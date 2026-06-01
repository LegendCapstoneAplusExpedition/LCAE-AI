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
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TypedDict

import numpy as np
import websockets
from dotenv import load_dotenv
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
    vad_threshold: float = float(_env("VAD_THRESHOLD", "0.02"))
    silence_duration_s: float = float(_env("VAD_SILENCE_DURATION_S", "0.8"))
    min_speech_duration_s: float = float(_env("VAD_MIN_SPEECH_DURATION_S", "0.3"))
    max_buffer_duration_s: float = float(_env("VAD_MAX_BUFFER_DURATION_S", "30.0"))

    # Whisper — HuggingFace 모델명(예: "base") 또는 로컬 faster-whisper 디렉터리 경로
    model: str = _env("ASR_MODEL", "base")
    device: str = _env("ASR_DEVICE", "auto")
    compute_type: str = _env("ASR_COMPUTE_TYPE", "auto")
    language: str = _env("ASR_LANGUAGE", "ko")
    beam_size: int = int(_env("ASR_BEAM_SIZE", "5"))

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
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        self.get_speech_timestamps = self.utils[0]
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
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        print(f"[Whisper] 모델 로딩: {config.model} | device={device} | compute={compute_type}")
        self.model = WhisperModel(
            config.model,
            device=device,
            compute_type=compute_type,
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
    ):
        self.config = config
        self.on_transcription = on_transcription
        self.transcriber = WhisperTranscriber(config)
        self.vad = SileroVAD(sample_rate=config.sample_rate, threshold=config.vad_threshold)
        self.buffer = SpeechBuffer(config=config)

    def push(self, chunk: np.ndarray):
        """float32 PCM 청크 1개를 입력. 발화 완성 시 콜백 호출 (별도 스레드에서)."""
        was_speaking = self.buffer._is_speaking
        is_speech = self.vad.is_speech(chunk)
        if is_speech and not was_speaking:
            print("\n[VAD] 음성 감지됨! (Speech Started)", flush=True)
        audio = self.buffer.push(chunk, is_speech)
        if audio is not None:
            print(f"[VAD] 발화 종료 → 전사 시작 ({len(audio) / self.config.sample_rate:.2f}s)", flush=True)
            t = threading.Thread(target=self._transcribe, args=(audio,), daemon=True)
            t.start()

    def _transcribe(self, audio: np.ndarray):
        result = self.transcriber.transcribe(audio)
        if result["text"].strip():
            self.on_transcription(result)


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
    ):
        self.host = host
        self.port = port
        self.config = config or PipelineConfig()
        self.on_transcription = on_transcription or (
            lambda r, ws: print(f"[ASR] {r['text']} (confidence={r['confidence']:.3f})")
        )

    async def _handle_client(self, websocket, path: str = "/"):
        print(f"[Server] 클라이언트 연결됨: {websocket.remote_address}")

        loop = asyncio.get_event_loop()
        
        # wrap the callback to include the websocket
        def callback_with_ws(result: TranscriptionResult):
            if asyncio.iscoroutinefunction(self.on_transcription):
                asyncio.run_coroutine_threadsafe(self.on_transcription(result, websocket), loop)
            else:
                self.on_transcription(result, websocket)

        core = ASRCore(
            self.config,
            on_transcription=callback_with_ws,
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
                    if data.get("type") == "end":
                        break
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
