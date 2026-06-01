"""
Capstone2026-1 — 통합 파이프라인 진입점

STT → LLM → TTS 순서로 연결되는 파이프라인입니다.

실행 예시:
    python main.py --mode mic
    python main.py --mode server --port 8765
    python main.py --mode client --ws-uri ws://localhost:8080/audio
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(".env.local", override=True)

_LISTENLIST_DIR = Path(__file__).parent / "pipeline" / "listenlist"


def _clear_session_files() -> None:
    _LISTENLIST_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ("transcriptions.jsonl", "chat.jsonl", "ai_outputs.jsonl", "ready_summary.json", "ready_question.json"):
        fpath = _LISTENLIST_DIR / fname
        fpath.write_text("", encoding="utf-8")
        print(f"[시작] {fname} 초기화 완료")

from pipeline.stt import MicrophoneASRTest, PipelineConfig, TranscriptionResult
from pipeline.tts import SynthesisResult, TTSConfig, TTSCore
from pipeline.listenlist import AIOutputList, ListenList


def build_pipeline(stt_config: PipelineConfig, tts_config: TTSConfig, topics: list[str] | None = None, on_synthesis=None):
    """STT → LLM → TTS 파이프라인을 조립하여 on_transcription 콜백을 반환합니다."""
    from pipeline.llm.chain.setup import mentor_setup
    from pipeline.llm.chain.graph import app as llm_app
    from langchain_core.messages import HumanMessage, AIMessage

    tts = TTSCore(tts_config, on_synthesis=on_synthesis or _on_synthesis)
    # 방송 세션 상태 — 호출 간 누적됨 (topics 기반으로 초기화)
    state = mentor_setup(topics or [])
    listen_list = ListenList()
    ai_outputs = AIOutputList()
    min_llm_confidence = float(os.getenv("ASR_MIN_LLM_CONFIDENCE", "0.55"))

    def on_transcription(result: TranscriptionResult) -> None:
        """STT 결과 수신 콜백 — ListenList 저장 → LLM → TTS 연결 지점"""
        print(f"[STT] {result['text']}  (conf={result['confidence']:.3f}, lang={result['language']})")

        # 1. ListenList에 저장
        entry = listen_list.append(result["text"], result["confidence"])
        all_entries = listen_list.read_all()
        print(f"[ListenList] 저장 ({entry['time']}, 누적 {len(all_entries)}건)")

        if result["confidence"] < min_llm_confidence:
            print(
                f"[LLM] STT 신뢰도 낮음({result['confidence']:.3f} < {min_llm_confidence:.2f}) "
                "→ LLM 호출 생략"
            )
            return

        # 2. 최근 발화 이력을 컨텍스트로 구성 (현재 항목 제외, 최대 10건)
        prior = [e for e in all_entries if e["time"] != entry["time"]][-10:]

        messages = []
        if prior:
            history = "\n".join(
                f"[{e['time']}ms] {e['text']} (신뢰도: {e['conf']:.2f})" for e in prior
            )
            messages.append(HumanMessage(content=f"[이전 발화 기록]\n{history}"))
        messages.append(HumanMessage(content=result["text"]))

        print(f"[LLM] ▶ 입력 전달: \"{result['text']}\"")

        # 3. 누적 state에 이번 messages를 반영하여 LLM 호출
        state["messages"] = messages
        state["silence_duration"] = result.get("silence_duration", 5.0)
        llm_result = llm_app.invoke(state)
        state.update(llm_result)

        msgs = state.get("messages", [])
        last_msg = msgs[-1] if msgs else None
        if isinstance(last_msg, AIMessage):
            print(f"[LLM] ◀ 최종 출력: \"{last_msg.content[:80]}{'...' if len(last_msg.content) > 80 else ''}\"")
            ai_outputs.append(
                mentor_text=result["text"],
                mentor_confidence=result["confidence"],
                state=state,
                ai_text=last_msg.content,
                spoken=True,
            )
            tts.synthesize(last_msg.content)
        else:
            ai_outputs.append(
                mentor_text=result["text"],
                mentor_confidence=result["confidence"],
                state=state,
                ai_text="",
                spoken=False,
            )
            print("[LLM] decision=wait → 발화 없음")

    return on_transcription


def _on_synthesis(result: SynthesisResult) -> None:
    """TTS 합성 완료 콜백 — 로그 출력 후 스피커로 즉시 재생"""
    print(f"[TTS] 합성 완료: {result['duration']:.2f}s ({len(result['audio'])} bytes) → 스피커 재생 중...")
    try:
        import numpy as np
        import sounddevice as sd
        pcm = np.frombuffer(result["audio"], dtype=np.int16).astype(np.float32) / 32767.0
        sd.play(pcm, samplerate=result["sample_rate"])
        sd.wait()
        print("[TTS] 재생 완료")
    except ImportError:
        print("[TTS] 경고: sounddevice 미설치 → 재생 생략 (pip install sounddevice)")
    except Exception as e:
        print(f"[TTS] 재생 오류: {e}")


def main() -> None:
    import argparse

    _clear_session_files()

    parser = argparse.ArgumentParser(description="Capstone2026-1 통합 파이프라인")
    parser.add_argument("--mode", choices=["mic", "server", "client"], default="mic",
                        help="mic: 마이크 테스트 / server: WebSocket 서버 / client: WebSocket 클라이언트")
    parser.add_argument("--host", default=os.getenv("WS_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WS_SERVER_PORT", "8765")))
    parser.add_argument("--ws-uri", default=os.getenv("WS_URI", "ws://localhost:8080/audio"))

    # STT 설정
    parser.add_argument("--model", default=os.getenv("ASR_MODEL", "base"),
                        help="Whisper 모델명 (tiny/base/small/medium/large-v3) 또는 로컬 경로")
    parser.add_argument("--language", default=os.getenv("ASR_LANGUAGE", "ko"))
    parser.add_argument("--device", default=os.getenv("ASR_DEVICE", "auto"), choices=["auto", "cpu", "cuda"])
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
        lang_code=args.language,
        ws_uri=args.tts_ws_uri,
    )

    if args.mode == "mic":
        on_transcription = build_pipeline(stt_config, tts_config)
        test = MicrophoneASRTest(stt_config, on_transcription=on_transcription)
        test.run(device=args.mic_device)

    elif args.mode == "server":
        import asyncio
        from pipeline.stt import RealtimeASRServer

        _ctx: dict = {"ws": None, "loop": None}

        def on_synthesis_server(result: SynthesisResult) -> None:
            ws = _ctx["ws"]
            loop = _ctx["loop"]
            if ws is not None and loop is not None and loop.is_running():
                asyncio.run_coroutine_threadsafe(ws.send(result["audio"]), loop)

        on_transcription = build_pipeline(stt_config, tts_config, on_synthesis=on_synthesis_server)

        def on_transcription_with_ws(result, websocket) -> None:
            _ctx["ws"] = websocket
            on_transcription(result)

        async def run_server():
            _ctx["loop"] = asyncio.get_running_loop()
            server = RealtimeASRServer(
                host=args.host, port=args.port,
                config=stt_config,
                on_transcription=on_transcription_with_ws,
            )
            await server.serve()

        asyncio.run(run_server())

    elif args.mode == "client":
        import asyncio
        from pipeline.stt import RealtimeASRPipeline
        on_transcription = build_pipeline(stt_config, tts_config)
        client = RealtimeASRPipeline(config=stt_config, on_transcription=on_transcription)
        try:
            asyncio.run(client.run())
        except KeyboardInterrupt:
            client.stop()


if __name__ == "__main__":
    main()
