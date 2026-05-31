"""
실제 방송 흐름 시뮬레이션 테스트.
각 utterance 사이에 state(context_summary, streaming_stage 등)가 누적됩니다.
MC는 멘토가 질문요청 / 정리요청 / 마무리를 명시적으로 요청할 때만 발화합니다.

실행:
    $env:PYTHONUTF8=1; uv run python -m pipeline.llm.chain.test_broadcast

출력:
    test_outputs/broadcast_YYYYMMDD_HHMMSS.jsonl   전체 로그 (모든 utterance)
    test_outputs/broadcast_YYYYMMDD_HHMMSS_tts.jsonl  TTS 큐 (speak=true 항목만)
"""

import json
import os
from datetime import datetime
from langchain_core.messages import HumanMessage

from pipeline.llm.chain.setup import mentor_setup
from pipeline.llm.chain.nodes import (
    preprocess_node,
    knowledge_search_node,
    analyze_write_node,
    decision_node,
    output_node,
)

# ── mc_type 매핑 ──────────────────────────────────────────────────────────────
_MC_TYPE = {
    "정리요청": "summary_recap",   # 누적 요약 읽기
    "질문요청": "question_relay",  # 청취자 질문 전달
    "마무리":   "closing",         # 클로징 멘트
}

# ── 방송 전문 ─────────────────────────────────────────────────────────────────
# 주제: 번아웃 없이 오래 가는 개발자 되는 법
# 의도적 테스트 케이스:
#   - 짧은 필러 ("어...") → LLM 스킵 확인
#   - "좋은 질문이에요" → 대기 (질문요청 오분류 방지)
#   - "다음 질문 받아볼게요" → 질문요청 (명확한 트리거)
#   - 정리요청은 방송 중반에 한 번만
#   - 마무리는 맨 마지막 발화 하나만
TRANSCRIPT = [
    # ── 오프닝 ───────────────────────────────────────────────
    {
        "text": "안녕하세요, 드라이빙 멘토링입니다. 오늘은 번아웃 없이 오래 가는 개발자가 되는 법을 이야기해볼게요.",
        "question_queue": [],
    },
    {
        "text": "개발자 10년을 버티는 사람이 드문 이유가 있어요. 기술은 빠르게 변하는데, 에너지 관리를 못 해서 스스로 나가떨어지는 경우가 많거든요.",
        "question_queue": [],
    },

    # ── 번아웃 원인 ──────────────────────────────────────────
    {
        "text": "번아웃은 갑자기 오지 않아요. 조금씩 쌓이다가 어느 순간 코드 보는 것 자체가 싫어지는 거예요.",
        "question_queue": [],
    },
    {
        # 짧은 필러 — LLM 스킵 대상
        "text": "어...",
        "question_queue": [],
    },
    {
        "text": "제가 직접 겪었을 때는요, 눈 뜨면 제일 먼저 드는 생각이 오늘 또 버그 고쳐야 하나 였어요. 그 시점이 이미 번아웃 초기 신호예요.",
        "question_queue": [],
    },

    # ── 전략 1: 작업 블록 분리 ──────────────────────────────
    {
        "text": "첫 번째 방법은 집중 블록과 커뮤니케이션 블록을 분리하는 거예요. 코딩하다 슬랙 보다 코딩하다 회의하다 이러면 뇌가 절대 회복을 못 해요.",
        "question_queue": [],
    },
    {
        "text": "저는 오전 두 시간은 알림을 다 끄고 코딩만 해요. 이것만으로도 퇴근 후 피로감이 확 달라졌어요.",
        "question_queue": [],
    },

    # ── 전략 2: 작은 완료 경험 ──────────────────────────────
    {
        "text": "두 번째는 작은 완료 경험을 의도적으로 만드는 거예요. 큰 기능만 계속 붙잡으면 몇 주째 완성한 게 없는 느낌이 들거든요.",
        "question_queue": [],
    },
    {
        "text": "하루에 하나씩, 배포 가능한 수준의 작은 것을 완성하세요. 그 완성 감각이 다음 날 에너지를 만들어줘요.",
        "question_queue": [],
    },

    # ── 전략 3: 자기 기록과 비교 ────────────────────────────
    {
        "text": "세 번째는 타인 비교 대신 어제의 나와 비교하는 거예요. GitHub 잔디나 팔로워 수 같은 건 번아웃 가속 페달이에요.",
        "question_queue": [],
    },
    {
        "text": "지난달보다 이번 달에 뭘 더 알게 됐는지, 그 차이에 집중하세요. 그게 실력이 늘고 있다는 증거예요.",
        "question_queue": [],
    },

    # ── 정리요청 (명확한 트리거) ─────────────────────────────
    {
        "text": "여기까지 말씀드린 내용, 잠깐 정리해드릴게요.",
        "question_queue": [],
    },

    # ── QnA 시작 (명확한 트리거) ─────────────────────────────
    {
        "text": "자, 청취자 질문 받아볼게요.",
        "question_queue": [
            "번아웃인지 슬럼프인지 어떻게 구분하나요?",
            "사이드 프로젝트가 번아웃에 도움이 되나요, 독이 되나요?",
            "팀 문화가 나쁠 때 개인이 할 수 있는 건 뭔가요?",
        ],
    },
    {
        # 질문에 대한 코멘트 — 대기로 분류돼야 함 (이전 오분류 케이스)
        "text": "좋은 질문이 들어왔네요.",
        "question_queue": [],
    },
    {
        # 명확한 질문요청 트리거
        "text": "다음 질문 받아볼게요.",
        "question_queue": [],
    },

    # ── 마무리 (단 하나) ─────────────────────────────────────
    {
        "text": "오늘 드라이빙 멘토링 여기서 마치겠습니다. 작업 블록 분리, 작은 완료 경험, 어제의 나와 비교하기. 이 세 가지 중 하나만 이번 주에 실천해보세요. 들어주셔서 감사합니다.",
        "question_queue": [],
    },
]


def _save_tts_queue(history: list[dict], out_dir: str, ts: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tts_queue_{ts}.jsonl")

    with open(path, "w", encoding="utf-8") as f:
        for row in history:
            if not row["speak"]:
                continue
            f.write(json.dumps({
                "no":      row["no"],
                "mc_type": row["mc_type"],
                "mc_text": row["mc_text"],
            }, ensure_ascii=False) + "\n")

    return path


def run():
    topics = ["번아웃 예방", "작업 블록 관리", "개발자 자기 관리"]
    state  = mentor_setup(topics)
    print(f"[LLM] provider=Ollama  model=driving-mentor")
    print(f"[Setup] topics={state['broadcast_topics']}")
    print(f"[Setup] streaming_stage={state['streaming_stage']}\n")

    history: list[dict] = []

    for i, utt in enumerate(TRANSCRIPT, 1):
        print(f"\n{'='*65}")
        print(f"  [{i:02d}/{len(TRANSCRIPT)}] \"{utt['text'][:50]}{'...' if len(utt['text']) > 50 else ''}\"")
        if utt["question_queue"]:
            print(f"  q_queue={len(utt['question_queue'])}개 유입")
        print('='*65)

        state["messages"]       = [HumanMessage(content=utt["text"])]
        state["question_queue"] = utt["question_queue"] if utt["question_queue"] else state["question_queue"]

        state.update(preprocess_node(state))
        state.update(knowledge_search_node(state))
        state.update(analyze_write_node(state))

        decision = decision_node(state)
        if decision == "speak":
            state.update(output_node(state))

        intent    = state.get("intent", "")
        mc_script = state.get("mc_script", "")
        spoke     = decision == "speak"

        history.append({
            "no":         i,
            "utterance":  utt["text"],
            "topic":      state.get("current_topic", ""),
            "intent":     intent,
            "stage":      state.get("streaming_stage", ""),
            "summary":    state.get("context_summary", ""),
            "speak":      spoke,
            "mc_type":    _MC_TYPE.get(intent) if spoke else None,
            "mc_text":    mc_script if spoke else "",
        })

    # ── 콘솔 요약 테이블 ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  방송 흐름 요약")
    print('='*65)
    print(f"{'No':>3}  {'intent':^8}  {'stage':^14}  {'speak':^5}  mc_text")
    print('-'*65)
    for h in history:
        preview = h["mc_text"][:38] + ("..." if len(h["mc_text"]) > 38 else "")
        print(f"{h['no']:>3}  {h['intent']:^8}  {h['stage']:^14}  {'O' if h['speak'] else '-':^5}  {preview}")

    print(f"\n[최종 누적 요약]\n{state.get('context_summary', '(없음)')}")

    # ── 파일 저장 ─────────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_outputs")
    tts_p   = _save_tts_queue(history, out_dir, ts)

    spoke_count = sum(1 for h in history if h["speak"])
    print(f"\n[Output] TTS 큐 저장 : {tts_p}")
    print(f"[Output] 발화 항목   : {spoke_count}건 / 전체 {len(history)}건")


if __name__ == "__main__":
    run()
