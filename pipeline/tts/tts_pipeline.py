"""
Real-time TTS Pipeline using gTTS (Google Text-to-Speech)

Architecture:
  text input → GTTSSynthesizer → SynthesisResult callback
                               → WebSocket 전송 (백엔드) / 로컬 스피커 재생 (테스트)

합성된 음성(PCM bytes + 메타데이터)은 on_synthesis 콜백으로 전달됩니다.
이후 처리(WebSocket 송신, 저장 등)는 호출자가 담당합니다.

엔진 교체: GTTSSynthesizer 대신 BaseTTSSynthesizer를 구현한 다른 클래스
(예: OpenAITTSSynthesizer, ClovaTTSSynthesizer)를 TTSCore에 주입하면 됩니다.
"""

import asyncio
import io
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, TypedDict

import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv()             # .env (공통 설정, git 커밋 O)
load_dotenv(".env.local", override=True)  # .env.local (민감 정보, git 커밋 X)


# ---------------------------------------------------------------------------
# 결과 타입
# ---------------------------------------------------------------------------

class SynthesisResult(TypedDict):
    audio: bytes       # PCM 16-bit mono raw bytes
    sample_rate: int   # 합성 샘플레이트 (기본: 24000)
    duration: float    # 합성 음성 길이 (초)
    language: str      # 언어 코드 (예: "ko")


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def _env(key: str, default: str) -> str:
    """환경변수 값을 반환. 없으면 default."""
    return os.environ.get(key, default)


@dataclass
class TTSConfig:
    # WebSocket — 합성 결과를 전송할 백엔드 URI
    ws_uri: str = field(default_factory=lambda: _env("TTS_WS_URI", "ws://localhost:8080/tts"))

    # gTTS 설정
    # lang_code: ISO 639-1 언어 코드 ('ko'=한국어, 'en'=영어, 'ja'=일본어 등)
    lang_code: str = field(default_factory=lambda: _env("TTS_LANG_CODE", "ko"))
    # tld: Google TTS 서버 도메인 (발음 변형 제어)
    #   'com' = 기본(미국식), 'co.uk' = 영국식, 'com.au' = 호주식
    tld: str = field(default_factory=lambda: _env("TTS_TLD", "com"))
    # slow: True이면 느린 속도로 합성
    slow: bool = field(default_factory=lambda: _env("TTS_SLOW", "false").lower() == "true")

    # 오디오 출력
    sample_rate: int = field(default_factory=lambda: int(_env("TTS_SAMPLE_RATE", "24000")))


# ---------------------------------------------------------------------------
# 추상 인터페이스 — 엔진 교체 지점
# ---------------------------------------------------------------------------

class BaseTTSSynthesizer:
    """
    TTS 엔진 교체를 위한 추상 인터페이스.

    외부 API(OpenAI TTS, Naver Clova 등)로 교체 시 이 클래스를 상속하여
    synthesize()만 구현하면 TTSCore를 변경할 필요 없습니다.
    """

    def synthesize(self, text: str) -> SynthesisResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# gTTS 구현체
# ---------------------------------------------------------------------------

class GTTSSynthesizer(BaseTTSSynthesizer):
    """
    gTTS (Google Text-to-Speech) 기반 합성기.

    필요 패키지:
        pip install gtts
        pip install pydub

    pydub의 MP3 디코딩에는 ffmpeg가 필요합니다:
        Windows: https://ffmpeg.org/download.html → PATH 등록
        Ubuntu:  sudo apt install ffmpeg
        macOS:   brew install ffmpeg

    gTTS는 Google TTS API를 사용하므로 인터넷 연결이 필요합니다.
    synthesize()는 thread-safe하지 않으므로 TTSCore가 직렬화(순차 스레드)합니다.
    """

    def __init__(self, config: TTSConfig):
        self.config = config
        try:
            from gtts import gTTS  # noqa: PLC0415 — 임포트 검증
            from pydub import AudioSegment  # noqa: PLC0415 — 임포트 검증
        except ImportError as e:
            raise ImportError(
                f"[TTS] 필수 패키지 누락: {e}\n"
                "  pip install gtts pydub\n"
                "  (MP3 디코딩을 위해 ffmpeg도 필요합니다)"
            ) from e

        print(f"[TTS] gTTS 초기화 | lang={config.lang_code} | tld={config.tld} | slow={config.slow}")

    def synthesize(self, text: str) -> SynthesisResult:
        """텍스트 → SynthesisResult (PCM 16-bit bytes + 메타데이터)"""
        from gtts import gTTS  # noqa: PLC0415
        from pydub import AudioSegment  # noqa: PLC0415

        # 1. gTTS로 MP3 합성 → BytesIO 버퍼
        tts = gTTS(text=text, lang=self.config.lang_code, tld=self.config.tld, slow=self.config.slow)
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)

        # 2. MP3 → PCM 변환 (pydub)
        segment = AudioSegment.from_mp3(mp3_buffer)
        segment = segment.set_frame_rate(self.config.sample_rate)  # 리샘플링
        segment = segment.set_channels(1)                           # 모노
        segment = segment.set_sample_width(2)                       # 16-bit

        pcm_bytes = segment.raw_data
        duration = len(segment) / 1000.0  # ms → 초

        return SynthesisResult(
            audio=pcm_bytes,
            sample_rate=self.config.sample_rate,
            duration=duration,
            language=self.config.lang_code,
        )


# ---------------------------------------------------------------------------
# TTSCore — 텍스트 입력 → SynthesisResult 콜백 (ASRCore 대칭 설계)
# ---------------------------------------------------------------------------

class TTSCore:
    """
    텍스트를 받아 TTS 합성 후 on_synthesis 콜백으로 결과를 전달합니다.
    합성은 별도 데몬 스레드에서 수행됩니다.

    사용 예:
        def handle(result: SynthesisResult):
            audio_bytes = result["audio"]  # PCM 16-bit raw bytes

        core = TTSCore(config, on_synthesis=handle)
        core.synthesize("안녕하세요")

    엔진 교체:
        synthesizer = MyCustomSynthesizer(config)
        core = TTSCore(config, on_synthesis=handle, synthesizer=synthesizer)
    """

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        on_synthesis: Optional[Callable[[SynthesisResult], None]] = None,
        synthesizer: Optional[BaseTTSSynthesizer] = None,
    ):
        self.config = config or TTSConfig()
        self.on_synthesis = on_synthesis or (
            lambda r: print(
                f"[TTS] 합성 완료: {r['duration']:.2f}s | {len(r['audio'])} bytes"
            )
        )
        self._synthesizer = synthesizer or GTTSSynthesizer(self.config)

    def synthesize(self, text: str):
        """텍스트 합성 요청. 결과는 on_synthesis 콜백으로 전달됩니다 (별도 스레드)."""
        t = threading.Thread(target=self._run, args=(text,), daemon=True)
        t.start()

    def _run(self, text: str):
        result = self._synthesizer.synthesize(text)
        self.on_synthesis(result)


# ---------------------------------------------------------------------------
# RealtimeTTSPipeline — 합성 결과를 WebSocket으로 백엔드에 전송
# ---------------------------------------------------------------------------

class RealtimeTTSPipeline:
    """
    TTSCore의 on_synthesis 결과를 WebSocket으로 백엔드에 전송합니다.
    연결 끊김 시 지수 백오프(1→2→4→8→…→60초)로 재연결합니다.

    사용 예:
        pipeline = RealtimeTTSPipeline(config)
        asyncio.run(pipeline.run())
        # 이후 다른 스레드에서:
        pipeline.synthesize("안녕하세요")
    """

    _RECONNECT_BASE_DELAY = 1.0
    _RECONNECT_MAX_DELAY = 60.0

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        on_synthesis: Optional[Callable[[SynthesisResult], None]] = None,
    ):
        self.config = config or TTSConfig()
        self._external_on_synthesis = on_synthesis
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._send_queue: Optional[asyncio.Queue] = None
        self._core: Optional[TTSCore] = None

    def _handle_synthesis(self, result: SynthesisResult):
        """합성 완료 콜백 — 스레드에서 asyncio 큐로 브리지"""
        if self._loop and self._send_queue:
            asyncio.run_coroutine_threadsafe(
                self._send_queue.put(result["audio"]),
                self._loop,
            )
        if self._external_on_synthesis:
            self._external_on_synthesis(result)

    def synthesize(self, text: str):
        """텍스트 합성 요청 — 완료 시 WebSocket으로 PCM 전송"""
        if self._core:
            self._core.synthesize(text)

    async def run(self):
        """WebSocket 연결 유지 루프. asyncio.run()으로 실행합니다."""
        self._loop = asyncio.get_event_loop()
        self._send_queue = asyncio.Queue()
        self._core = TTSCore(self.config, on_synthesis=self._handle_synthesis)

        delay = self._RECONNECT_BASE_DELAY

        while not self._stop_event.is_set():
            try:
                print(f"[TTS Pipeline] 백엔드 연결 시도: {self.config.ws_uri}")
                async with websockets.connect(self.config.ws_uri) as ws:
                    print("[TTS Pipeline] 연결 완료. 합성 대기 중...")
                    delay = self._RECONNECT_BASE_DELAY  # 연결 성공 시 딜레이 초기화
                    while not self._stop_event.is_set():
                        try:
                            audio = await asyncio.wait_for(
                                self._send_queue.get(), timeout=1.0
                            )
                            await ws.send(audio)
                        except asyncio.TimeoutError:
                            continue
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                if self._stop_event.is_set():
                    break
                print(f"[TTS Pipeline] 연결 끊김: {e}. {delay:.0f}초 후 재연결...")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._RECONNECT_MAX_DELAY)

    def stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# SpeakerTTSTest — 로컬 스피커 직접 재생 (MicrophoneASRTest 대칭)
# ---------------------------------------------------------------------------

class SpeakerTTSTest:
    """
    백엔드 없이 로컬 스피커로 TTS 합성 결과를 직접 재생하는 테스트 클래스.
    표준 입력으로 텍스트를 받아 합성 후 즉시 재생합니다.

    필요 패키지:
        pip install sounddevice

    사용 예:
        # 키보드 입력 → 합성 → 스피커 재생
        tester = SpeakerTTSTest()
        tester.run()

        # 콜백으로 합성 결과 받기
        def save_audio(result: SynthesisResult): ...
        tester = SpeakerTTSTest(on_synthesis=save_audio)
        tester.run()
    """

    def __init__(
        self,
        config: Optional[TTSConfig] = None,
        on_synthesis: Optional[Callable[[SynthesisResult], None]] = None,
    ):
        self.config = config or TTSConfig()
        self._external_on_synthesis = on_synthesis
        self._core: Optional[TTSCore] = None
        self._stop_event = threading.Event()
        self._synthesis_history: list[dict] = []

    def _handle_synthesis(self, result: SynthesisResult):
        """합성 완료 시 스피커 재생 + 히스토리 저장"""
        import sounddevice as sd  # noqa: PLC0415

        entry = {
            "duration": result["duration"],
            "language": result["language"],
            "timestamp": time.strftime("%H:%M:%S"),
            "bytes": len(result["audio"]),
        }
        self._synthesis_history.append(entry)

        # PCM 16-bit → float32 변환 후 재생
        pcm = np.frombuffer(result["audio"], dtype=np.int16).astype(np.float32) / 32767.0
        sd.play(pcm, samplerate=result["sample_rate"])
        sd.wait()  # 재생 완료까지 대기

        if self._external_on_synthesis:
            self._external_on_synthesis(result)
        else:
            print(f"\n  [재생 완료] {entry['timestamp']}  {entry['duration']:.2f}s\n")

    @staticmethod
    def list_devices():
        """사용 가능한 오디오 출력 디바이스 목록 출력"""
        import sounddevice as sd  # noqa: PLC0415

        devices = sd.query_devices()
        print("\n[SpeakerTest] 사용 가능한 오디오 출력 디바이스:")
        print(f"  {'ID':>3}  {'이름':<45}  {'최대 출력 채널':>6}  {'샘플레이트':>10}")
        print("  " + "-" * 75)
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                print(f"  {i:>3}  {d['name']:<45}  {d['max_output_channels']:>6}  {d['default_samplerate']:>10.0f}")
        default_idx = sd.default.device[1]
        print(f"\n  * 현재 기본 디바이스: {default_idx} - {devices[default_idx]['name']}")

    def run(self, device: Optional[int] = None):
        """
        키보드 입력 텍스트를 합성하여 스피커로 재생합니다.

        Args:
            device: sounddevice 출력 디바이스 ID (None = 시스템 기본 스피커)
        """
        import sounddevice as sd  # noqa: PLC0415

        self._stop_event.clear()
        self._synthesis_history.clear()
        self._core = TTSCore(self.config, on_synthesis=self._handle_synthesis)

        # 출력 디바이스 설정
        if device is not None:
            sd.default.device[1] = device

        print("\n" + "=" * 60)
        print("  Real-time gTTS — 스피커 테스트")
        print(f"  lang: {self.config.lang_code} | tld: {self.config.tld} | slow: {self.config.slow}")
        print("  텍스트 입력 후 Enter → 합성 후 스피커 재생")
        if self._external_on_synthesis:
            print("  콜백 모드: 합성 결과를 on_synthesis으로 전달")
        print("  Ctrl+C 로 종료")
        print("=" * 60 + "\n")

        try:
            while not self._stop_event.is_set():
                try:
                    text = input("입력> ").strip()
                except EOFError:
                    break
                if text:
                    self._core.synthesize(text)
                    time.sleep(0.1)  # 스레드 시작 대기
        except KeyboardInterrupt:
            print("\n\n[SpeakerTest] 종료 요청.")
        finally:
            self._print_summary()

    def stop(self):
        self._stop_event.set()

    def _print_summary(self):
        print("\n" + "=" * 60)
        print(f"  합성 결과 요약 (총 {len(self._synthesis_history)}건)")
        print("=" * 60)
        for i, entry in enumerate(self._synthesis_history, 1):
            print(f"  [{i:>2}] {entry['timestamp']}  {entry['duration']:.2f}s  ({entry['bytes']} bytes)")
        print("=" * 60)

    def get_synthesis_history(self) -> list[dict]:
        """지금까지 합성된 결과 리스트 반환"""
        return list(self._synthesis_history)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time gTTS Pipeline")
    parser.add_argument("--mode", choices=["speaker", "client"], default="speaker",
                        help="speaker: 로컬 스피커 테스트 / client: WebSocket으로 백엔드에 PCM 전송")
    parser.add_argument("--ws-uri", default="ws://localhost:8080/tts")
    parser.add_argument("--lang", default="ko",
                        help="ISO 언어 코드 (예: ko, en, ja)")
    parser.add_argument("--tld", default="com",
                        help="Google TTS 도메인 (com=기본, co.uk=영국식 등)")
    parser.add_argument("--slow", action="store_true",
                        help="느린 속도로 합성")
    parser.add_argument("--speaker-device", type=int, default=None,
                        help="스피커 디바이스 ID (--mode speaker 전용, 생략 시 기본 스피커)")
    parser.add_argument("--list-devices", action="store_true",
                        help="사용 가능한 스피커 목록 출력 후 종료")
    args = parser.parse_args()

    if args.list_devices:
        SpeakerTTSTest.list_devices()
        exit(0)

    config = TTSConfig(
        ws_uri=args.ws_uri,
        lang_code=args.lang,
        tld=args.tld,
        slow=args.slow,
    )

    if args.mode == "speaker":
        tester = SpeakerTTSTest(config=config)
        tester.run(device=args.speaker_device)
    else:
        pipeline = RealtimeTTSPipeline(config=config)
        try:
            asyncio.run(pipeline.run())
        except KeyboardInterrupt:
            pipeline.stop()
