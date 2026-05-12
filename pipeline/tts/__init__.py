"""
TTS (Text-to-Speech) pipeline — Coqui TTS 기반 실시간 음성 합성

엔진 교체: BaseTTSSynthesizer를 상속한 구현체를 TTSCore에 주입하면
TTSCore 코드 변경 없이 외부 API(OpenAI TTS, Naver Clova 등)로 교체 가능합니다.
"""

from .tts_pipeline import (
    BaseTTSSynthesizer,
    KokoroTTSSynthesizer,
    RealtimeTTSPipeline,
    SpeakerTTSTest,
    SynthesisResult,
    TTSConfig,
    TTSCore,
)

__all__ = [
    "SynthesisResult",
    "TTSConfig",
    "BaseTTSSynthesizer",
    "KokoroTTSSynthesizer",
    "TTSCore",
    "RealtimeTTSPipeline",
    "SpeakerTTSTest",
]
