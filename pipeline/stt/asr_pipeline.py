"""
Real-time ASR Pipeline using Whisper

Architecture:
  Audio source --> VAD --> SpeechBuffer --> WhisperTranscriber --> TranscriptionResult callback

전사된 텍스트(+ 신뢰도)는 on_transcription 콜백으로 전달됩니다.
이후 처리(LLM 전달, 저장, WebSocket 송신 등)는 호출자가 담당합니다.
"""

import asyncio
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TypedDict

import numpy as np
import websockets
from dotenv import load_dotenv

# ── 네이티브 라이브러리 교착 예방 (faster_whisper/torch import 보다 먼저) ──────────
# Windows에서 ctranslate2(OpenMP)와 torch(MKL/OpenMP)가 같은 프로세스에 로드되면
# libiomp5md.dll이 중복 초기화되어, 워커 스레드에서 모델을 호출할 때 간헐적으로
# 교착/크래시가 발생한다("잘 되다가 어느 순간 전사가 멈춤"의 유력 원인).
# 이 플래그는 중복 OpenMP 런타임을 허용해 그 교착을 회피한다.
# 네이티브 라이브러리 로딩 전에 설정해야 효과가 있으므로 import보다 위에 둔다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from faster_whisper import WhisperModel

load_dotenv()
load_dotenv(".env.local", override=True)

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)

# ---------------------------------------------------------------------------
# 결과 타입
# ---------------------------------------------------------------------------

class TranscriptionResult(TypedDict):
    text: str           # 전사된 텍스트
    confidence: float   # 세그먼트 평균 log-prob → exp 정규화 [0, 1]
    language: str       # 감지/설정된 언어 코드 (예: "ko")


@dataclass
class PipelineConfig:
    # WebSocket
    ws_uri: str = _env("WS_URI", "ws://localhost:8080/audio")
    result_uri: str = "ws://localhost:8080/asr"

    # 오디오
    sample_rate: int = int(_env("AUDIO_SAMPLE_RATE", "16000"))
    channels: int = int(_env("AUDIO_CHANNELS", "1"))
    chunk_duration_ms: int = int(_env("AUDIO_CHUNK_DURATION_MS", "30"))

    # VAD
    # Silero-VAD는 0~1 확률을 반환하므로 임계값 기본값은 0.5 (에너지 VAD의 0.02가 아님)
    vad_threshold: float = float(_env("VAD_THRESHOLD", "0.5"))
    silence_duration_s: float = float(_env("VAD_SILENCE_DURATION_S", "0.8"))
    min_speech_duration_s: float = float(_env("VAD_MIN_SPEECH_DURATION_S", "0.3"))
    max_buffer_duration_s: float = float(_env("VAD_MAX_BUFFER_DURATION_S", "30.0"))

    # Whisper — HuggingFace 모델명(예: "base") 또는 로컬 faster-whisper 디렉터리 경로
    model: str = _env("ASR_MODEL", "base")
    device: str = _env("ASR_DEVICE", "auto")
    compute_type: str = _env("ASR_COMPUTE_TYPE", "auto")
    language: str = _env("ASR_LANGUAGE", "ko")
    beam_size: int = int(_env("ASR_BEAM_SIZE", "5"))
    # ctranslate2 내부 스레드 수. 0(auto)은 환경에 따라 과다 생성되어 OpenMP 경합을
    # 키울 수 있어 명시 고정한다. num_workers는 항상 1(단일 스레드 호출 보장).
    cpu_threads: int = int(_env("ASR_CPU_THREADS", "4"))

    # 안정성(전사 워커 적체/교착 방지)
    # 전사 대기열 최대 적재 건수. 초과 시 가장 오래된 발화를 버려 실시간성을 지킨다.
    max_pending_audio: int = int(_env("ASR_MAX_PENDING_AUDIO", "8"))
    # 단일 전사가 이 시간(초)을 넘기면 워커 교착으로 보고 경고를 남긴다(워치독).
    transcribe_timeout_s: float = float(_env("ASR_TRANSCRIBE_TIMEOUT_S", "60"))

    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_ms / 1000)

    @property
    def silence_chunks(self) -> int:
        return int(self.silence_duration_s * 1000 / self.chunk_duration_ms)

    @property
    def max_buffer_chunks(self) -> int:
        return int(self.max_buffer_duration_s * 1000 / self.chunk_duration_ms)


# ---------------------------------------------------------------------------
# VAD (에너지 기반 — Silero VAD로 교체 가능)
# ---------------------------------------------------------------------------

class EnergyVAD:
    """단순 RMS 에너지 기반 VAD. Silero-VAD로 교체하려면 SileroVAD 클래스 사용."""

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def is_speech(self, pcm_float32: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(pcm_float32 ** 2))
        return rms > self.threshold


class SileroVAD:
    """Silero-VAD 래퍼 (더 정확한 VAD가 필요할 때 사용)"""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        import torch
        # torch.hub.load(master 브랜치)는 네트워크/버전 변동에 취약해 로딩이 멈추거나
        # 깨질 수 있어, pip 패키지(silero-vad)로 고정해서 오프라인·재현 가능하게 로드한다.
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()  # onnx=False(기본), 네트워크 불필요
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.torch = torch

    def is_speech(self, pcm_float32: np.ndarray) -> bool:
        # Silero VAD requires fixed chunk sizes: 512 samples @ 16kHz, 256 @ 8kHz
        required = 512 if self.sample_rate == 16000 else 256
        if len(pcm_float32) != required:
            padded = np.zeros(required, dtype=np.float32)
            n = min(len(pcm_float32), required)
            padded[:n] = pcm_float32[:n]
            pcm_float32 = padded
        #energyVAD
        tensor = self.torch.from_numpy(pcm_float32)
        speech_prob = self.model(tensor, self.sample_rate).item()
        return speech_prob >= self.threshold


# ---------------------------------------------------------------------------
# Whisper 전사기
# ---------------------------------------------------------------------------

class WhisperTranscriber:
    def __init__(self, config: PipelineConfig):
        device = config.device
        compute_type = config.compute_type

        # device/compute_type 자동 설정
        if device == "auto":
            try:
                import ctranslate2
                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        print(
            f"[Whisper] 모델 로딩: {config.model} | device={device} | "
            f"compute={compute_type} | cpu_threads={config.cpu_threads}"
        )
        self.model = WhisperModel(
            config.model,
            device=device,
            compute_type=compute_type,
            cpu_threads=config.cpu_threads,
            num_workers=1,  # 동시 호출 금지(모델 thread-safety 보장)
        )
        self.config = config
        print("[Whisper] 모델 로딩 완료")

    def transcribe(self, pcm_float32: np.ndarray) -> TranscriptionResult:
        """float32 PCM numpy 배열 → TranscriptionResult (텍스트 + 신뢰도 + 언어)"""
        segments_gen, info = self.model.transcribe(
            pcm_float32,
            beam_size=self.config.beam_size,
            language=self.config.language,
            vad_filter=True,          # faster-whisper 내장 VAD 필터
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        segments = list(segments_gen)  # generator 소비

        text = " ".join(seg.text.strip() for seg in segments)

        # confidence: 세그먼트 길이 가중 평균 exp(avg_logprob) → [0, 1]
        if segments:
            total_duration = sum(seg.end - seg.start for seg in segments)
            if total_duration > 0:
                confidence = sum(
                    math.exp(seg.avg_logprob) * (seg.end - seg.start)
                    for seg in segments
                ) / total_duration
            else:
                confidence = math.exp(segments[0].avg_logprob)
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0

        return TranscriptionResult(
            text=text,
            confidence=confidence,
            language=info.language or self.config.language,
        )


# ---------------------------------------------------------------------------
# 오디오 버퍼 + VAD 상태 머신
# ---------------------------------------------------------------------------

@dataclass
class SpeechBuffer:
    config: PipelineConfig
    _chunks: list = field(default_factory=list)
    _silence_count: int = 0
    _is_speaking: bool = False

    def push(self, chunk: np.ndarray, is_speech: bool) -> Optional[np.ndarray]:
        """
        청크를 받아 버퍼에 추가.
        발화가 끝났거나 버퍼가 꽉 찼으면 전사할 오디오 배열 반환, 아니면 None.
        """
        if is_speech:
            self._is_speaking = True
            self._silence_count = 0
            self._chunks.append(chunk)
        else:
            if self._is_speaking:
                self._silence_count += 1
                self._chunks.append(chunk)  # 무음도 버퍼에 포함 (자연스러운 종료)

        # 발화 종료 조건
        speech_ended = (
            self._is_speaking
            and self._silence_count >= self.config.silence_chunks
        )
        # 버퍼 오버플로우 조건
        buffer_full = len(self._chunks) >= self.config.max_buffer_chunks

        if speech_ended or buffer_full:
            audio = self._flush()
            duration = len(audio) / self.config.sample_rate
            if duration >= self.config.min_speech_duration_s:
                return audio
            # 너무 짧으면 무시
        return None

    def _flush(self) -> np.ndarray:
        audio = np.concatenate(self._chunks) if self._chunks else np.array([], dtype=np.float32)
        self._chunks.clear()
        self._silence_count = 0
        self._is_speaking = False
        return audio


# ---------------------------------------------------------------------------
# 공통 ASR 코어 (오디오 → TranscriptionResult 변환만 담당)
# ---------------------------------------------------------------------------

class ASRCore:
    """
    VAD + SpeechBuffer + WhisperTranscriber를 묶은 핵심 ASR 유닛.

    오디오 청크를 push()하면 발화가 완성될 때마다 on_transcription 콜백으로
    TranscriptionResult(텍스트 + 신뢰도 + 언어)를 전달합니다.

    사용 예:
        def handle(result: TranscriptionResult):
            print("전사:", result["text"], "신뢰도:", result["confidence"])

        core = ASRCore(config, on_transcription=handle)
        core.push(chunk_float32)
    """

    def __init__(
        self,
        config: PipelineConfig,
        on_transcription: Callable[[TranscriptionResult], None],
        on_speech_start: Optional[Callable[[], None]] = None,
    ):
        self.config = config
        self.on_transcription = on_transcription
        self.on_speech_start = on_speech_start  # 음성 시작 감지 시 호출 (바지인 인터럽트용)
        self.transcriber = WhisperTranscriber(config)
        self.vad = SileroVAD(sample_rate=config.sample_rate, threshold=config.vad_threshold)
        self.buffer = SpeechBuffer(config=config)

        # ── 동시성 모델 ──────────────────────────────────────────────────────
        # 발화마다 새 스레드를 띄우면 (1) thread-safe하지 않은 Whisper 모델을
        # 동시 호출해 교착되고, (2) 다운스트림(LLM 그래프 + TTS)이 공유 state를
        # 경합하며 전사가 중간에 멈춘다.
        # 따라서 두 단계를 각각 "단일 워커 스레드"로 직렬화한다.
        #   - 전사 워커: Whisper 호출을 한 스레드에서만 → thread-safe 보장
        #   - 후처리 워커: 느린 LLM/TTS를 전사와 분리 → 전사가 막히지 않음
        self._audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._result_queue: "queue.Queue[Optional[TranscriptionResult]]" = queue.Queue()
        self._closed = False
        self._dropped_count = 0

        # 워치독: 전사 워커가 응답 없이 멈추면(교착) 조용한 정지 대신 경고를 남긴다.
        # _inflight_since는 현재 진행 중인 transcribe()의 시작 시각(monotonic), 없으면 None.
        self._inflight_since: Optional[float] = None

        self._transcribe_worker = threading.Thread(
            target=self._transcribe_loop, daemon=True, name="asr-transcribe"
        )
        self._downstream_worker = threading.Thread(
            target=self._downstream_loop, daemon=True, name="asr-downstream"
        )
        self._watchdog_worker = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="asr-watchdog"
        )
        self._transcribe_worker.start()
        self._downstream_worker.start()
        self._watchdog_worker.start()
        # 바지인 인터럽트 페이싱: 단발성 노이즈에 끊기지 않도록
        # 누적 발화 길이(BARGE_IN_MIN_SPEECH_MS) 이상일 때만 발화 구간당 1회 트리거.
        # VAD가 발화 중에도 깜빡일 수 있으므로 짧은 공백은 무시하고,
        # silence_chunks 만큼 공백이 이어져 발화 구간이 끝났을 때만 리셋한다.
        _min_ms = float(_env("BARGE_IN_MIN_SPEECH_MS", "350"))
        self._barge_min_chunks = max(1, int(_min_ms / config.chunk_duration_ms))
        self._speech_run = 0       # 현재 구간 누적 발화 청크 수
        self._gap_run = 0          # 연속 비음성 청크 수
        self._barge_fired = False  # 현재 발화 구간에서 이미 트리거했는지

    def push(self, chunk: np.ndarray):
        """float32 PCM 청크 1개를 입력. 발화 완성 시 전사 대기열에 적재 (논블로킹)."""
        was_speaking = self.buffer._is_speaking
        is_speech = self.vad.is_speech(chunk)
        if is_speech and not was_speaking:
            print("\n[VAD] 음성 감지됨! (Speech Started)", flush=True)

        # 보수적 바지인 트리거
        if is_speech:
            self._speech_run += 1
            self._gap_run = 0
            if (not self._barge_fired
                    and self._speech_run >= self._barge_min_chunks
                    and self.on_speech_start):
                self._barge_fired = True
                print(f"[VAD] 바지인 트리거 (연속 발화 {self._speech_run * self.config.chunk_duration_ms}ms)", flush=True)
                try:
                    self.on_speech_start()
                except Exception:
                    pass
        else:
            self._gap_run += 1
            # 발화 구간이 실제로 끝났을 때(공백이 충분히 길 때)만 리셋
            if self._gap_run >= self.config.silence_chunks:
                self._speech_run = 0
                self._barge_fired = False

        audio = self.buffer.push(chunk, is_speech)
        if audio is not None:
            # 백로그가 한계를 넘으면 가장 오래된 발화부터 버린다. 전사 워커가 입력
            # 속도를 못 따라가거나 일시 정체될 때 메모리가 무한정 늘지 않게 하고,
            # 오래된(이미 흘러간) 발화보다 최신 발화를 우선 처리해 실시간성을 지킨다.
            while self._audio_queue.qsize() >= self.config.max_pending_audio:
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.task_done()
                    self._dropped_count += 1
                except queue.Empty:
                    break
            backlog = self._audio_queue.qsize()
            note = f", 폐기 누적 {self._dropped_count}건" if self._dropped_count else ""
            print(
                f"[VAD] 발화 종료 → 전사 대기열 적재 "
                f"({len(audio) / self.config.sample_rate:.2f}s, 대기 {backlog}건{note})",
                flush=True,
            )
            self._audio_queue.put(audio)

    def _transcribe_loop(self):
        """전사 전용 워커 — Whisper 모델을 단일 스레드에서만 호출한다."""
        while True:
            audio = self._audio_queue.get()
            try:
                if audio is None:  # 종료 신호
                    break
                self._inflight_since = time.monotonic()  # 워치독 감시 시작
                result = self.transcriber.transcribe(audio)
                if result["text"].strip():
                    self._result_queue.put(result)
            except Exception as e:
                print(f"[ASR] 전사 오류: {e}", flush=True)
            finally:
                self._inflight_since = None  # 워치독 감시 해제
                self._audio_queue.task_done()

    def _watchdog_loop(self):
        """전사 워커 감시 — transcribe()가 비정상적으로 오래 멈추면 경고를 남긴다.

        Whisper 모델은 thread-safe하지 않아 다른 스레드에서 강제로 끊거나 재호출할 수
        없다(동시 호출 시 교착). 또한 이 프로세스가 죽으면 Node가 에이전트를 재시작 없이
        정리하므로 자동 종료도 부적절하다. 따라서 워치독은 '조용한 정지'를 '명확한 경고'로
        바꾸는 역할만 한다 — 로그(stdout)는 Node가 수집하므로 즉시 원인 파악이 가능하다.
        """
        timeout = self.config.transcribe_timeout_s
        warned = False
        while not self._closed:
            time.sleep(2.0)
            since = self._inflight_since
            if since is None:
                warned = False
                continue
            elapsed = time.monotonic() - since
            if elapsed >= timeout and not warned:
                warned = True
                print(
                    f"[ASR][WATCHDOG] 전사가 {elapsed:.0f}s째 응답 없음 — 워커 교착 의심. "
                    f"대기열 {self._audio_queue.qsize()}건 적체 중. "
                    f"지속되면 AI 에이전트 재시작이 필요합니다.",
                    flush=True,
                )

    def _downstream_loop(self):
        """후처리 전용 워커 — LLM 그래프/TTS(느린 작업)를 단일 스레드에서 순차 실행한다."""
        while True:
            result = self._result_queue.get()
            try:
                if result is None:  # 종료 신호
                    break
                self.on_transcription(result)
            except Exception as e:
                print(f"[ASR] 후처리(LLM/TTS) 오류: {e}", flush=True)
            finally:
                self._result_queue.task_done()

    def close(self):
        """워커 스레드를 정리한다. (연결 종료/테스트 종료 시 호출)"""
        if self._closed:
            return
        self._closed = True
        self._audio_queue.put(None)
        self._result_queue.put(None)


# ---------------------------------------------------------------------------
# WebSocket 모드 — 오디오 수신 후 TranscriptionResult 콜백
# ---------------------------------------------------------------------------

class RealtimeASRPipeline:
    """
    백엔드 WebSocket에 클라이언트로 연결해 오디오를 수신하고,
    전사 결과를 on_transcription 콜백으로 반환합니다.
    연결 끊김 시 지수 백오프(1→2→4→8→…→60초)로 재연결합니다.
    """

    _RECONNECT_BASE_DELAY = 1.0
    _RECONNECT_MAX_DELAY = 60.0

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None,
    ):
        self.config = config or PipelineConfig()
        self.on_transcription = on_transcription or (
            lambda r: print(f"[ASR] {r['text']} (confidence={r['confidence']:.3f})")
        )
        self._stop_event = threading.Event()

    @staticmethod
    def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        return pcm_int16.astype(np.float32) / 32768.0

    async def run(self):
        core = ASRCore(self.config, self.on_transcription)
        delay = self._RECONNECT_BASE_DELAY
        frame_bytes = self.config.chunk_samples * np.dtype(np.int16).itemsize

        while not self._stop_event.is_set():
            try:
                print(f"[Pipeline] 백엔드 연결 시도: {self.config.ws_uri}")
                async with websockets.connect(self.config.ws_uri) as ws:
                    print("[Pipeline] 연결 완료. 오디오 수신 중...")
                    delay = self._RECONNECT_BASE_DELAY  # 연결 성공 시 딜레이 초기화
                    pending_raw = bytearray()
                    async for message in ws:
                        if isinstance(message, bytes):
                            raw = message
                        else:
                            import base64
                            raw = base64.b64decode(json.loads(message)["audio"])

                        pending_raw.extend(raw)
                        while len(pending_raw) >= frame_bytes:
                            frame = bytes(pending_raw[:frame_bytes])
                            del pending_raw[:frame_bytes]
                            chunk = self._bytes_to_float32(frame)
                            core.push(chunk)
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                if self._stop_event.is_set():
                    break
                print(f"[Pipeline] 연결 끊김: {e}. {delay:.0f}초 후 재연결...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._RECONNECT_MAX_DELAY)

    def stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# WebSocket 서버 모드 — 백엔드가 연결해 오는 구조
# ---------------------------------------------------------------------------

class RealtimeASRServer:
    """
    백엔드가 WebSocket 클라이언트로 이 서버에 연결하는 구조.
    전사 결과는 on_transcription 콜백으로 반환합니다.
    on_transcription이 None이면 콘솔에 출력합니다.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        config: Optional[PipelineConfig] = None,
        on_transcription: Optional[Callable[[TranscriptionResult, Optional[websockets.WebSocketServerProtocol]], None]] = None,
        session_factory: Optional[Callable[[websockets.WebSocketServerProtocol], Callable[[TranscriptionResult], None]]] = None,
    ):
        self.host = host
        self.port = port
        self.config = config or PipelineConfig()
        # session_factory(websocket) -> on_transcription(result)
        #   연결마다 독립 파이프라인(state/listen_list/TTS)을 생성해 동시 접속 시
        #   상태 경합을 원천 차단한다. 지정하면 on_transcription보다 우선한다.
        self.session_factory = session_factory
        self.on_transcription = on_transcription or (
            lambda r, ws: print(f"[ASR] {r['text']} (confidence={r['confidence']:.3f})")
        )

    async def _handle_client(self, websocket, path: str = "/"):
        print(f"[Server] 클라이언트 연결됨: {websocket.remote_address}")

        loop = asyncio.get_event_loop()

        # 연결별 콜백 구성
        session_stop_tts = None
        session_reset = None
        if self.session_factory is not None:
            # 이 연결만을 위한 독립 파이프라인 (state/listen_list/TTS 격리)
            session_on_transcription = self.session_factory(websocket)
            # 세션이 노출한 TTS 취소 훅(바지인 시 진행/대기 합성 폐기)
            session_stop_tts = getattr(session_on_transcription, "stop_tts", None)
            # 세션 상태 초기화 + 오프닝 발화 훅(재소환 시 백엔드의 reset 신호로 호출)
            session_reset = getattr(session_on_transcription, "reset_session", None)

            def callback_with_ws(result: TranscriptionResult):
                session_on_transcription(result)
        else:
            # 레거시: 단일 공유 콜백 (result, websocket) 시그니처
            def callback_with_ws(result: TranscriptionResult):
                if asyncio.iscoroutinefunction(self.on_transcription):
                    asyncio.run_coroutine_threadsafe(self.on_transcription(result, websocket), loop)
                else:
                    self.on_transcription(result, websocket)

        # 바지인: 멘토 음성이 감지되면
        #   (1) 이 프로세스의 TTSCore 합성 큐를 취소하고(다음 문장이 안 나가게)
        #   (2) 백엔드에 interrupt 신호를 보내 ffmpeg 버퍼를 비우게 한다.
        def notify_speech_start():
            if session_stop_tts is not None:
                try:
                    session_stop_tts()
                except Exception:
                    pass
            try:
                asyncio.run_coroutine_threadsafe(
                    websocket.send(json.dumps({"type": "interrupt"})), loop
                )
            except Exception:
                pass

        core = ASRCore(
            self.config,
            on_transcription=callback_with_ws,
            on_speech_start=notify_speech_start,
        )

        chunk_count = 0
        pending_raw = bytearray()
        frame_bytes = self.config.chunk_samples * np.dtype(np.int16).itemsize
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    raw = message
                else:
                    data = json.loads(message)
                    mtype = data.get("type")
                    if mtype == "end":
                        break
                    # 재소환: 백엔드가 브리지를 새로 연결하면 세션 상태를 초기화하고
                    # 오프닝 멘트를 다시 발화한다. (오디오 프레임이 아니므로 여기서 종료)
                    if mtype == "reset":
                        if session_reset is not None:
                            print("[Server] reset 신호 수신 → 세션 초기화")
                            await loop.run_in_executor(None, session_reset)
                        continue
                    import base64
                    raw = base64.b64decode(data["audio"])

                # 데이터 수신 시각화 (100청크마다 출력하여 너무 시끄럽지 않게 함)
                chunk_count += 1
                if chunk_count % 100 == 0:
                    print(".", end="", flush=True)

                pending_raw.extend(raw)
                while len(pending_raw) >= frame_bytes:
                    frame = bytes(pending_raw[:frame_bytes])
                    del pending_raw[:frame_bytes]
                    pcm_int16 = np.frombuffer(frame, dtype=np.int16)
                    chunk = pcm_int16.astype(np.float32) / 32768.0
                    await loop.run_in_executor(None, core.push, chunk)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            core.close()  # 워커 스레드 누수 방지

        print(f"\n[Server] 클라이언트 연결 종료: {websocket.remote_address}")

    async def serve(self):
        print(f"[Server] ASR 서버 시작: ws://{self.host}:{self.port}")
        async with websockets.serve(self._handle_client, self.host, self.port):
            await asyncio.Future()  # 무한 대기


# ---------------------------------------------------------------------------
# 마이크 직접 테스트 (백엔드 없이)
# ---------------------------------------------------------------------------

class MicrophoneASRTest:
    """
    백엔드 연결 없이 실제 마이크 입력으로 ASR 파이프라인을 테스트하는 클래스.

    on_transcription 콜백을 지정하면 TranscriptionResult를 외부로 전달할 수 있습니다.
    지정하지 않으면 콘솔에 출력합니다.

    필요 패키지:
        pip install sounddevice

    사용 예:
        # 콘솔 출력만
        tester = MicrophoneASRTest()
        tester.run()

        # 콜백으로 결과 받기 (LLM 연동 등)
        def send_to_llm(result: TranscriptionResult): ...
        tester = MicrophoneASRTest(on_transcription=send_to_llm)
        tester.run()
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_transcription: Optional[Callable[[TranscriptionResult], None]] = None,
    ):
        self.config = config or PipelineConfig()
        self._on_transcription = on_transcription  # None이면 기본 출력 사용
        self._core: Optional[ASRCore] = None
        self._stop_event = threading.Event()
        self._transcript_history: list[dict] = []

    def _handle_transcription(self, result: TranscriptionResult):
        """전사 완료 시 호출 — 히스토리 저장 후 콜백 또는 기본 출력"""
        entry = {
            "text": result["text"],
            "confidence": result["confidence"],
            "language": result["language"],
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self._transcript_history.append(entry)

        if self._on_transcription:
            self._on_transcription(result)
        else:
            print(f"\n[{entry['timestamp']}] >> {entry['text']}  (conf={entry['confidence']:.3f})\n")

    # ── 디바이스 확인 ────────────────────────────────────────────────────────

    @staticmethod
    def list_devices():
        """사용 가능한 오디오 입력 디바이스 목록 출력"""
        import sounddevice as sd
        devices = sd.query_devices()
        print("\n[MicTest] 사용 가능한 오디오 입력 디바이스:")
        print(f"  {'ID':>3}  {'이름':<45}  {'최대 입력 채널':>6}  {'샘플레이트':>10}")
        print("  " + "-" * 75)
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                print(f"  {i:>3}  {d['name']:<45}  {d['max_input_channels']:>6}  {d['default_samplerate']:>10.0f}")
        default_idx = sd.default.device[0]
        print(f"\n  * 현재 기본 디바이스: {default_idx} - {devices[default_idx]['name']}")

    # ── 마이크 스트림 콜백 ────────────────────────────────────────────────────

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info, status):
        """sounddevice 스트림 콜백 — 메인 스레드와 분리된 오디오 스레드에서 호출됨"""
        if status:
            print(f"[MicTest] 스트림 상태: {status}", flush=True)

        chunk = indata[:, 0].copy()  # mono float32rr

        # VAD 상태 표시
        indicator = "█" if self._core.vad.is_speech(chunk) else "·"
        print(indicator, end="", flush=True)

        # ASRCore.push → 발화 완성 시 _handle_transcription 호출
        self._core.push(chunk)

    # ── 실행 ────────────────────────────────────────────────────────────────

    def run(self, device: Optional[int] = None, duration_s: Optional[float] = None):
        """
        마이크 입력으로 실시간 ASR 실행.

        Args:
            device: sounddevice 디바이스 ID (None = 시스템 기본 마이크)
            duration_s: 녹음 시간 제한(초). None이면 Ctrl+C 누를 때까지 무한 실행.
        """
        import sounddevice as sd

        self._stop_event.clear()
        self._transcript_history.clear()
        self._core = ASRCore(self.config, on_transcription=self._handle_transcription)

        print("\n" + "=" * 60)
        print("  Real-time Whisper ASR — 마이크 테스트")
        print(f"  모델: {self.config.model} | 언어: {self.config.language}")
        print(f"  VAD 임계값: {self.config.vad_threshold} | 묵음 판정: {self.config.silence_duration_s}s")
        if self._on_transcription:
            print("  콜백 모드: 전사 결과를 on_transcription으로 전달")
        print("  [█ = 발화  · = 묵음]  Ctrl+C 로 종료")
        print("=" * 60 + "\n")

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=self.config.chunk_samples,
                device=device,
                callback=self._audio_callback,
            ):
                if duration_s:
                    time.sleep(duration_s)
                else:
                    while not self._stop_event.is_set():
                        time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n[MicTest] 종료 요청.")
        finally:
            if self._core is not None:
                self._core.close()
            self._print_summary()

    def stop(self):
        self._stop_event.set()

    # ── 결과 요약 ────────────────────────────────────────────────────────────

    def _print_summary(self):
        print("\n" + "=" * 60)
        print(f"  전사 결과 요약 (총 {len(self._transcript_history)}건)")
        print("=" * 60)
        for i, entry in enumerate(self._transcript_history, 1):
            print(f"  [{i:>2}] {entry['timestamp']}  {entry['text']}  (conf={entry['confidence']:.3f})")
        print("=" * 60)

    def get_transcript_history(self) -> list[dict]:
        """지금까지 전사된 결과 리스트 반환"""
        return list(self._transcript_history)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Whisper ASR Pipeline")
    parser.add_argument("--mode", choices=["client", "server", "mic"], default="mic",
                        help="mic: 마이크 테스트 / server: 백엔드 연결 수신 / client: 백엔드에 연결")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ws-uri", default="ws://localhost:8080/audio")
    parser.add_argument("--model", default="base",
                        help="HuggingFace 모델명(tiny/base/small/medium/large-v3) 또는 로컬 경로")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mic-device", type=int, default=None,
                        help="마이크 디바이스 ID (--mode mic 전용, 생략 시 기본 마이크)")
    parser.add_argument("--list-devices", action="store_true",
                        help="사용 가능한 마이크 목록 출력 후 종료")
    args = parser.parse_args()

    if args.list_devices:
        MicrophoneASRTest.list_devices()
        exit(0)

    config = PipelineConfig(
        ws_uri=args.ws_uri,
        model=args.model,
        language=args.language,
        device=args.device,
    )

    if args.mode == "mic":
        tester = MicrophoneASRTest(config=config)
        tester.run(device=args.mic_device)
    elif args.mode == "server":
        server = RealtimeASRServer(host=args.host, port=args.port, config=config)
        asyncio.run(server.serve())
    else:
        pipeline = RealtimeASRPipeline(config=config)
        try:
            asyncio.run(pipeline.run())
        except KeyboardInterrupt:
            pipeline.stop()
