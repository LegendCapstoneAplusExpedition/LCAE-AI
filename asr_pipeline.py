"""
Real-time ASR Pipeline using Whisper

Architecture:
  Audio source --> VAD --> SpeechBuffer --> WhisperTranscriber --> str (text)

전사된 텍스트는 on_transcription 콜백으로 전달됩니다.
이후 처리(LLM 전달, 저장, WebSocket 송신 등)는 호출자가 담당합니다.
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import websockets
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # WebSocket
    ws_uri: str = "ws://localhost:8080/audio"   # 백엔드 WebSocket 주소
    result_uri: str = "ws://localhost:8080/asr"  # 결과 전송 주소 (같은 소켓 사용 가능)

    # 오디오
    sample_rate: int = 16000          # Whisper 기본 sample rate
    channels: int = 1
    chunk_duration_ms: int = 30       # 한 청크 길이 (ms)

    # VAD (Silero-VAD 기반 간단 에너지 VAD 대체 가능)
    vad_threshold: float = 0.02       # RMS 에너지 임계값
    silence_duration_s: float = 0.8   # 이 시간 동안 조용하면 발화 종료로 판단
    min_speech_duration_s: float = 0.3  # 최소 발화 길이 (너무 짧은 노이즈 무시)
    max_buffer_duration_s: float = 30   # 최대 버퍼 길이 (강제 전사)

    # Whisper
    model_size: str = "base"          # tiny / base / small / medium / large-v3
    device: str = "auto"              # auto / cpu / cuda
    compute_type: str = "auto"        # auto / int8 / float16
    language: str = "ko"              # None이면 자동 감지
    beam_size: int = 5

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
        )
        self.get_speech_timestamps = self.utils[0]
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._h = None
        self._c = None
        import torch
        self.torch = torch

    def is_speech(self, pcm_float32: np.ndarray) -> bool:
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

        print(f"[Whisper] 모델 로딩: {config.model_size} | device={device} | compute={compute_type}")
        self.model = WhisperModel(
            config.model_size,
            device=device,
            compute_type=compute_type,
        )
        self.config = config
        print("[Whisper] 모델 로딩 완료")

    def transcribe(self, pcm_float32: np.ndarray) -> str:
        """float32 PCM numpy 배열 -> 전사 텍스트"""
        segments, info = self.model.transcribe(
            pcm_float32,
            beam_size=self.config.beam_size,
            language=self.config.language,
            vad_filter=True,          # faster-whisper 내장 VAD 필터
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text


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
# 공통 ASR 코어 (오디오 → 텍스트 변환만 담당)
# ---------------------------------------------------------------------------

class ASRCore:
    """
    VAD + SpeechBuffer + WhisperTranscriber를 묶은 핵심 ASR 유닛.

    오디오 청크를 push()하면 발화가 완성될 때마다 on_transcription 콜백으로
    전사된 텍스트(str)를 전달합니다. 이후 처리는 호출자가 담당합니다.

    사용 예:
        def handle(text: str):
            print("전사:", text)

        core = ASRCore(config, on_transcription=handle)
        core.push(chunk_float32)
    """

    def __init__(
        self,
        config: PipelineConfig,
        on_transcription: Callable[[str], None],
    ):
        self.config = config
        self.on_transcription = on_transcription
        self.transcriber = WhisperTranscriber(config)
        self.vad = EnergyVAD(threshold=config.vad_threshold)
        self.buffer = SpeechBuffer(config=config)

    def push(self, chunk: np.ndarray):
        """float32 PCM 청크 1개를 입력. 발화 완성 시 콜백 호출 (별도 스레드에서)."""
        is_speech = self.vad.is_speech(chunk)
        audio = self.buffer.push(chunk, is_speech)
        if audio is not None:
            t = threading.Thread(target=self._transcribe, args=(audio,), daemon=True)
            t.start()

    def _transcribe(self, audio: np.ndarray):
        text = self.transcriber.transcribe(audio)
        if text.strip():
            self.on_transcription(text.strip())


# ---------------------------------------------------------------------------
# WebSocket 모드 — 오디오 수신 후 텍스트 콜백
# ---------------------------------------------------------------------------

class RealtimeASRPipeline:
    """
    백엔드 WebSocket에 클라이언트로 연결해 오디오를 수신하고,
    전사 결과를 on_transcription 콜백으로 반환합니다.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
    ):
        self.config = config or PipelineConfig()
        self.on_transcription = on_transcription or (lambda text: print(f"[ASR] {text}"))
        self._stop_event = threading.Event()

    @staticmethod
    def _bytes_to_float32(raw_bytes: bytes) -> np.ndarray:
        pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        return pcm_int16.astype(np.float32) / 32768.0

    async def run(self):
        core = ASRCore(self.config, self.on_transcription)

        print(f"[Pipeline] 백엔드 연결 시도: {self.config.ws_uri}")
        async with websockets.connect(self.config.ws_uri) as ws:
            print("[Pipeline] 연결 완료. 오디오 수신 중...")
            async for message in ws:
                if isinstance(message, bytes):
                    chunk = self._bytes_to_float32(message)
                else:
                    import base64
                    raw = base64.b64decode(json.loads(message)["audio"])
                    chunk = self._bytes_to_float32(raw)
                core.push(chunk)

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
        on_transcription: Optional[Callable[[str], None]] = None,
    ):
        self.host = host
        self.port = port
        self.config = config or PipelineConfig()
        self.on_transcription = on_transcription or (lambda text: print(f"[ASR] {text}"))

    async def _handle_client(self, websocket, path: str = "/"):
        print(f"[Server] 클라이언트 연결: {websocket.remote_address}")

        loop = asyncio.get_event_loop()
        core = ASRCore(
            self.config,
            on_transcription=self.on_transcription,
        )

        async for message in websocket:
            if isinstance(message, bytes):
                raw = message
            else:
                data = json.loads(message)
                if data.get("type") == "end":
                    break
                import base64
                raw = base64.b64decode(data["audio"])

            pcm_int16 = np.frombuffer(raw, dtype=np.int16)
            chunk = pcm_int16.astype(np.float32) / 32768.0
            # push는 내부적으로 별도 스레드에서 전사 실행
            await loop.run_in_executor(None, core.push, chunk)

        print(f"[Server] 클라이언트 종료: {websocket.remote_address}")

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

    on_transcription 콜백을 지정하면 전사 텍스트를 외부로 전달할 수 있습니다.
    지정하지 않으면 콘솔에 출력합니다.

    필요 패키지:
        pip install sounddevice

    사용 예:
        # 콘솔 출력만
        tester = MicrophoneASRTest()
        tester.run()

        # 콜백으로 텍스트 받기 (LLM 연동 등)
        def send_to_llm(text: str): ...
        tester = MicrophoneASRTest(on_transcription=send_to_llm)
        tester.run()
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        on_transcription: Optional[Callable[[str], None]] = None,
    ):
        self.config = config or PipelineConfig()
        self._on_transcription = on_transcription  # None이면 기본 출력 사용
        self._core: Optional[ASRCore] = None
        self._stop_event = threading.Event()
        self._transcript_history: list[dict] = []

    def _handle_transcription(self, text: str):
        """전사 완료 시 호출 — 히스토리 저장 후 콜백 또는 기본 출력"""
        entry = {"text": text, "timestamp": time.strftime("%H:%M:%S")}
        self._transcript_history.append(entry)

        if self._on_transcription:
            self._on_transcription(text)
        else:
            print(f"\n[{entry['timestamp']}] >> {text}\n")

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

        chunk = indata[:, 0].copy()  # mono float32

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
        print(f"  모델: {self.config.model_size} | 언어: {self.config.language}")
        print(f"  VAD 임계값: {self.config.vad_threshold} | 묵음 판정: {self.config.silence_duration_s}s")
        if self._on_transcription:
            print("  콜백 모드: 전사 텍스트를 on_transcription으로 전달")
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
            print(f"  [{i:>2}] {entry['timestamp']}  {entry['text']}")
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
                        choices=["tiny", "base", "small", "medium", "large-v3"])
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
        model_size=args.model,
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
