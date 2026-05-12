"""
Capstone2026-1 — 통합 파이프라인 진입점

STT → LLM → TTS 순서로 연결되는 파이프라인입니다.
현재는 STT → TTS가 구현되어 있으며, LLM은 추후 사이에 추가됩니다.

실행 예시:
    python main.py --mode mic
    python main.py --mode server --port 8765
    python main.py --mode client --ws-uri ws://localhost:8080/audio
"""

import sys

from pipeline.stt import MicrophoneASRTest, PipelineConfig, TranscriptionResult
from pipeline.tts import SynthesisResult, TTSConfig, TTSCore


def build_pipeline(stt_config: PipelineConfig, tts_config: TTSConfig):
    """STT → (LLM) → TTS 파이프라인을 조립하여 on_transcription 콜백을 반환합니다."""

    tts = TTSCore(tts_config, on_synthesis=_on_synthesis)

    def on_transcription(result: TranscriptionResult) -> None:
        """STT 결과 수신 콜백 — LLM → TTS 연결 지점"""
        print(f"[STT] {result['text']}  (conf={result['confidence']:.3f}, lang={result['language']})")

        # TODO: LLM 처리 연결 시 아래 주석 해제
        # from pipeline.llm.chain.graph import app as llm_app
        # from pipeline.llm.chain.state import AgentState
        # from langchain_core.messages import HumanMessage
        # llm_state = AgentState(messages=[HumanMessage(content=result["text"])], ...)
        # llm_result = llm_app.invoke(llm_state)
        # response = llm_result["messages"][-1].content
        # tts.synthesize(response)

        tts.synthesize(result["text"])  # 현재: STT → TTS 직결 (LLM 미연결)

    return on_transcription


def _on_synthesis(result: SynthesisResult) -> None:
    """TTS 합성 완료 콜백"""
    print(f"[TTS] 합성 완료: {result['duration']:.2f}s ({len(result['audio'])} bytes)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Capstone2026-1 통합 파이프라인")
    parser.add_argument("--mode", choices=["mic", "server", "client"], default="mic",
                        help="mic: 마이크 테스트 / server: WebSocket 서버 / client: WebSocket 클라이언트")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ws-uri", default="ws://localhost:8080/audio")

    # STT 설정
    parser.add_argument("--model", default="base",
                        help="Whisper 모델명 (tiny/base/small/medium/large-v3) 또는 로컬 경로")
    parser.add_argument("--language", default="ko")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mic-device", type=int, default=None,
                        help="마이크 디바이스 ID (--mode mic 전용, 생략 시 기본 마이크)")
    parser.add_argument("--list-devices", action="store_true",
                        help="사용 가능한 마이크 목록 출력 후 종료")

    # TTS 설정
    parser.add_argument("--tts-model", default="tts_models/ko/css10/vits",
                        help="Coqui TTS 모델명 또는 로컬 경로")
    parser.add_argument("--tts-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--tts-ws-uri", default="ws://localhost:8080/tts",
                        help="TTS 합성 결과를 전송할 백엔드 WebSocket URI")

    args = parser.parse_args()

    if args.list_devices:
        MicrophoneASRTest.list_devices()
        sys.exit(0)

    stt_config = PipelineConfig(
        model=args.model,
        language=args.language,
        device=args.device,
    )
    tts_config = TTSConfig(
        model=args.tts_model,
        language=args.language,
        device=args.tts_device,
        ws_uri=args.tts_ws_uri,
    )

    on_transcription = build_pipeline(stt_config, tts_config)

    if args.mode == "mic":
        test = MicrophoneASRTest(stt_config, on_transcription=on_transcription)
        test.run(device=args.mic_device)

    elif args.mode == "server":
        import asyncio
        from pipeline.stt import RealtimeASRServer
        server = RealtimeASRServer(host=args.host, port=args.port,
                                   config=stt_config, on_transcription=on_transcription)
        asyncio.run(server.serve())

    elif args.mode == "client":
        import asyncio
        from pipeline.stt import RealtimeASRPipeline
        client = RealtimeASRPipeline(config=stt_config, on_transcription=on_transcription)
        try:
            asyncio.run(client.run())
        except KeyboardInterrupt:
            client.stop()


if __name__ == "__main__":
    main()
