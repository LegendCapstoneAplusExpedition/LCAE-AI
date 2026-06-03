"""
Run the LLM graph against a temporary broadcast transcript.

This bypasses STT/TTS and checks:
- general intent/topic JSON
- bridge speech
- question request path from chat.jsonl
- summary request path from ready_summary.json
- closing speech

Example:
    python llm_training/tests/check_llm_pipeline.py
    python llm_training/tests/check_llm_pipeline.py --model driving-mentor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
LISTENLIST_DIR = ROOT_DIR / "pipeline" / "listenlist"
DEFAULT_TRANSCRIPT = ROOT_DIR / "llm_training" / "tests" / "sample_broadcast.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "llm_training" / "tests" / "llm_pipeline_outputs.jsonl"
DEFAULT_SUMMARY_OUTPUT = ROOT_DIR / "llm_training" / "tests" / "summary_outputs.jsonl"
SESSION_FILES = [
    "transcriptions.jsonl",
    "chat.jsonl",
    "ai_outputs.jsonl",
    "ready_summary.json",
    "ready_question.json",
]


def load_transcript(path: Path) -> list[str]:
    utterances: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            utterances.append(line)
            continue
        text = str(data.get("text", "")).strip()
        if not text:
            raise ValueError(f"{path}:{line_no}: text is empty")
        utterances.append(text)
    if not utterances:
        raise ValueError(f"No transcript utterances found: {path}")
    return utterances


@contextmanager
def isolated_listenlist_files(enabled: bool = True):
    if not enabled:
        yield
        return

    LISTENLIST_DIR.mkdir(parents=True, exist_ok=True)
    backup: dict[Path, bytes | None] = {}
    for name in SESSION_FILES:
        path = LISTENLIST_DIR / name
        backup[path] = path.read_bytes() if path.exists() else None
        path.write_text("", encoding="utf-8")
    try:
        yield
    finally:
        for path, content in backup.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)


def seed_runtime_files(broadcast_id: str) -> None:
    chat = {
        "time": "2026-06-03T12:00:00.000Z",
        "broadcast_id": broadcast_id,
        "user_id": "test-user-1",
        "username": "민지",
        "message": "사이드 프로젝트 고객 인터뷰는 몇 명 정도 해보면 좋을까요?",
        "is_question": True,
        "used": False,
    }
    (LISTENLIST_DIR / "chat.jsonl").write_text(
        json.dumps(chat, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_ready_summary_payload() -> dict[str, Any]:
    path = LISTENLIST_DIR / "ready_summary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def read_ready_summary() -> str:
    return str(read_ready_summary_payload().get("summary", "")).strip()


def append_summary_output(
    path: Path,
    turn: int,
    requested_source_count: int,
    payload: dict[str, Any],
    recent_weight_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "turn": turn,
        "requested_source_count": requested_source_count,
        "source_count": payload.get("source_count", requested_source_count),
        "previous_source_count": payload.get("previous_source_count", 0),
        "new_count": payload.get("new_count", requested_source_count),
        "recent_weight_count": recent_weight_count,
        "summary": str(payload.get("summary", "")).strip(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_summary_request(text: str) -> bool:
    return any(keyword in text for keyword in ("요약", "정리"))


def append_transcription(text: str) -> list[dict[str, Any]]:
    path = LISTENLIST_DIR / "transcriptions.jsonl"
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "conf": 1.0,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


def invoke_pipeline(args: argparse.Namespace, utterances: list[str]) -> list[dict[str, Any]]:
    os.environ["BROADCAST_ID"] = args.broadcast_id
    os.environ["AI_BRIDGE_COOLDOWN_S"] = str(args.cooldown)
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    from langchain_core.messages import AIMessage, HumanMessage
    from pipeline.llm.chain.graph import app
    from pipeline.llm.chain.setup import mentor_setup
    from pipeline.listenlist.listen_list import ListenList, SUMMARY_INTERVAL

    state = mentor_setup(args.topic)
    outputs: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    listen_list = ListenList()

    for idx, text in enumerate(utterances, 1):
        entries = append_transcription(text)
        should_refresh_summary = (
            len(entries) > 0
            and (
                len(entries) % args.summary_interval == 0
                or is_summary_request(text)
            )
        )
        if args.generate_summary and should_refresh_summary:
            summary_entries = entries[:-1] if is_summary_request(text) and len(entries) > 1 else entries
            if summary_entries:
                print(f"\n[SummaryTest] llm_summary 호출: {len(summary_entries)}개 전사")
                summary_payload = listen_list._summarize_to_ready(summary_entries)
                if not summary_payload:
                    summary_payload = read_ready_summary_payload()
                new_count = int(summary_payload.get("new_count", len(summary_entries)) or 0)
                append_summary_output(
                    args.summary_output,
                    idx,
                    len(summary_entries),
                    summary_payload,
                    min(new_count, SUMMARY_INTERVAL),
                )

        prior = history[-args.history :]
        messages = []
        if prior:
            history_text = "\n".join(
                f"[{row['turn']:02d}] {row['mentor']}" for row in prior
            )
            messages.append(HumanMessage(content=f"[이전 발화 기록]\n{history_text}"))
        messages.append(HumanMessage(content=text))

        state["messages"] = messages
        state["silence_duration"] = args.silence
        if args.cooldown == 0:
            state["last_ai_speech_ts"] = 0.0

        started = time.time()
        result = app.invoke(state)
        state.update(result)
        elapsed = time.time() - started

        final_messages = state.get("messages", [])
        last_message = final_messages[-1] if final_messages else None
        ai_text = last_message.content if isinstance(last_message, AIMessage) else ""

        row = {
            "turn": idx,
            "mentor": text,
            "topic": state.get("current_topic", ""),
            "intent": state.get("intent", ""),
            "stage": state.get("streaming_stage", ""),
            "mc_script": state.get("mc_script", ""),
            "ai_text": ai_text,
            "spoken": bool(ai_text),
            "ready_summary": read_ready_summary(),
            "elapsed_s": round(elapsed, 2),
        }
        outputs.append(row)
        history.append(row)
        print_turn(row)

    return outputs


def print_turn(row: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print(f"[{row['turn']:02d}] mentor: {row['mentor']}")
    print(
        f"topic={row['topic']} | intent={row['intent']} | "
        f"stage={row['stage']} | elapsed={row['elapsed_s']}s"
    )
    if row["spoken"]:
        print(f"AI: {row['ai_text']}")
    else:
        print("AI: <wait>")
    if row.get("ready_summary"):
        print(f"ready_summary: {row['ready_summary']}")


def write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\noutput written: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LLM pipeline with a temporary transcript")
    parser.add_argument("--transcript", type=Path, default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--topic", action="append", default=["사이드 프로젝트 MVP", "고객 인터뷰", "번아웃 예방"])
    parser.add_argument("--model", default="", help="Override LLM_MODEL, e.g. driving-mentor")
    parser.add_argument("--broadcast-id", default="llm-pipeline-test")
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--silence", type=float, default=5.0)
    parser.add_argument("--cooldown", type=float, default=0.0, help="Bridge cooldown seconds for this test")
    parser.add_argument("--summary-interval", type=int, default=3)
    parser.add_argument("--no-generate-summary", dest="generate_summary", action="store_false")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--keep-runtime-files", action="store_true")
    parser.set_defaults(generate_summary=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utterances = load_transcript(args.transcript)
    if args.summary_output and not args.keep_runtime_files:
        args.summary_output.unlink(missing_ok=True)
    with isolated_listenlist_files(enabled=not args.keep_runtime_files):
        seed_runtime_files(args.broadcast_id)
        outputs = invoke_pipeline(args, utterances)
    write_output(args.output, outputs)


if __name__ == "__main__":
    main()
