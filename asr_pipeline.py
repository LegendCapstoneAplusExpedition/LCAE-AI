"""
Real-time ASR Pipeline using Whisper
백엔드로부터 WebSocket으로 오디오를 받아 실시간 텍스트화

Architecture:
  Backend --[WebSocket]--> AudioReceiver --> VAD --> WhisperTranscriber --> Backend
"""

import asyncio
import json
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

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
# 메인 파이프라인
# ---------------------------------------------------------------------------

class RealtimeASRPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.transcriber = WhisperTranscriber(self.config)
        self.vad = EnergyVAD(threshold=self.config.vad_threshold)
        self.speech_buffer = SpeechBuffer(config=self.config)

        self._audio_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

    # ── 오디오 수신 ─────────────────────────────────────────────────────────

    def _bytes_to_float32(self, raw_bytes: bytes) -> np.ndarray:
        """
        백엔드에서 받은 raw bytes -> float32 PCM.
        백엔드 포맷에 따라 int16 또는 float32로 변경하세요.
        """
        pcm_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        return pcm_int16.astype(np.float32) / 32768.0

    async def _receive_audio(self, websocket):
        """WebSocket에서 오디오 청크를 받아 큐에 적재"""
        async for message in websocket:
            if isinstance(message, bytes):
                chunk = self._bytes_to_float32(message)
            else:
                # JSON 래핑된 경우: {"audio": "<base64>"}
                import base64
                data = json.loads(message)
                raw = base64.b64decode(data["audio"])
                chunk = self._bytes_to_float32(raw)

            self._audio_queue.put(chunk)

    # ── VAD + 버퍼링 (별도 스레드) ──────────────────────────────────────────

    def _vad_worker(self):
        while not self._stop_event.is_set():
            try:
                chunk = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            is_speech = self.vad.is_speech(chunk)
            audio = self.speech_buffer.push(chunk, is_speech)

            if audio is not None:
                self._transcribe_worker(audio)

    def _transcribe_worker(self, audio: np.ndarray):
        """동기 전사 호출 (VAD 스레드 내에서 실행)"""
        start = time.time()
        text = self.transcriber.transcribe(audio)
        elapsed = time.time() - start

        if text.strip():
            result = {
                "text": text.strip(),
                "duration_s": round(len(audio) / self.config.sample_rate, 2),
                "latency_s": round(elapsed, 2),
                "timestamp": time.time(),
            }
            print(f"[ASR] ({elapsed:.2f}s) {text.strip()}")
            self._result_queue.put(result)

    # ── 결과 전송 ───────────────────────────────────────────────────────────

    async def _send_results(self, websocket):
        """전사 결과를 WebSocket으로 백엔드에 전송"""
        loop = asyncio.get_event_loop()
        while not self._stop_event.is_set():
            try:
                result = await loop.run_in_executor(
                    None, lambda: self._result_queue.get(timeout=0.1)
                )
                await websocket.send(json.dumps(result, ensure_ascii=False))
            except queue.Empty:
                continue
            except websockets.exceptions.ConnectionClosed:
                break

    # ── 실행 ────────────────────────────────────────────────────────────────

    async def run(self):
        """파이프라인 실행 (단일 WebSocket에서 수신+송신)"""
        vad_thread = threading.Thread(target=self._vad_worker, daemon=True)
        vad_thread.start()

        print(f"[Pipeline] 백엔드 연결 시도: {self.config.ws_uri}")
        async with websockets.connect(self.config.ws_uri) as ws:
            print("[Pipeline] 연결 완료. 오디오 수신 중...")
            await asyncio.gather(
                self._receive_audio(ws),
                self._send_results(ws),
            )

    def stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# 백엔드가 클라이언트로 연결해 오는 경우 (서버 모드)
# ---------------------------------------------------------------------------

class RealtimeASRServer:
    """
    백엔드가 WebSocket 클라이언트로 이 서버에 연결하는 구조.
    백엔드 → 오디오 → ASR서버 → 텍스트 → 백엔드
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, config: Optional[PipelineConfig] = None):
        self.host = host
        self.port = port
        self.config = config or PipelineConfig()
        self.transcriber = WhisperTranscriber(self.config)

    async def _handle_client(self, websocket, path: str = "/"):
        print(f"[Server] 클라이언트 연결: {websocket.remote_address}")
        vad = EnergyVAD(threshold=self.config.vad_threshold)
        buffer = SpeechBuffer(config=self.config)

        async def bytes_to_float32(raw: bytes) -> np.ndarray:
            pcm_int16 = np.frombuffer(raw, dtype=np.int16)
            return pcm_int16.astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()

        async for message in websocket:
            if isinstance(message, bytes):
                chunk = await bytes_to_float32(message)
            else:
                data = json.loads(message)
                if data.get("type") == "end":
                    break
                import base64
                raw = base64.b64decode(data["audio"])
                chunk = await bytes_to_float32(raw)

            is_speech = vad.is_speech(chunk)
            audio = buffer.push(chunk, is_speech)

            if audio is not None:
                # 전사는 executor에서 실행 (블로킹 방지)
                text = await loop.run_in_executor(
                    None, self.transcriber.transcribe, audio
                )
                if text.strip():
                    result = {
                        "text": text.strip(),
                        "duration_s": round(len(audio) / self.config.sample_rate, 2),
                        "timestamp": time.time(),
                    }
                    await websocket.send(json.dumps(result, ensure_ascii=False))
                    print(f"[Server] 전송: {text.strip()}")

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

    필요 패키지:
        pip install sounddevice

    사용 예:
        tester = MicrophoneASRTest()
        tester.list_devices()          # 사용 가능한 마이크 목록 출력
        tester.run(device=None)        # None이면 시스템 기본 마이크 사용
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.transcriber = WhisperTranscriber(self.config)
        self.vad = EnergyVAD(threshold=self.config.vad_threshold)
        self.speech_buffer = SpeechBuffer(config=self.config)
        self._stop_event = threading.Event()
        self._transcript_history: list[dict] = []

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

        # indata shape: (frames, channels) → mono float32 1D
        chunk = indata[:, 0].copy()

        is_speech = self.vad.is_speech(chunk)
        audio = self.speech_buffer.push(chunk, is_speech)

        # VAD 상태 표시 (터미널 시각화)
        indicator = "█" if is_speech else "·"
        print(indicator, end="", flush=True)

        if audio is not None:
            print()  # VAD 표시줄 줄바꿈
            # 전사는 별도 스레드에서 (콜백을 블로킹하지 않기 위해)
            t = threading.Thread(target=self._transcribe_and_print, args=(audio,), daemon=True)
            t.start()

    def _transcribe_and_print(self, audio: np.ndarray):
        duration = len(audio) / self.config.sample_rate
        print(f"\n[MicTest] 전사 중... ({duration:.1f}초 분량)")
        start = time.time()
        text = self.transcriber.transcribe(audio)
        elapsed = time.time() - start

        if text.strip():
            entry = {
                "text": text.strip(),
                "duration_s": round(duration, 2),
                "latency_s": round(elapsed, 2),
                "timestamp": time.strftime("%H:%M:%S"),
            }
            self._transcript_history.append(entry)
            print(f"[{entry['timestamp']}] ({elapsed:.2f}s) >> {text.strip()}\n")
        else:
            print("[MicTest] (인식 결과 없음)\n")

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

        print("\n" + "=" * 60)
        print("  Real-time Whisper ASR — 마이크 테스트")
        print(f"  모델: {self.config.model_size} | 언어: {self.config.language}")
        print(f"  VAD 임계값: {self.config.vad_threshold} | 묵음 판정: {self.config.silence_duration_s}s")
        print("  [█ = 발화  · = 묵음]  Ctrl+C 로 종료")
        print("=" * 60 + "\n")

        chunk_samples = self.config.chunk_samples

        try:
            with sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=chunk_samples,
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
            print(f"        (발화 {entry['duration_s']}s → 전사 {entry['latency_s']}s)")
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
