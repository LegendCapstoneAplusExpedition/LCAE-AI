"""
세션(방송)별 파일 경로 리졸버.

여러 방송을 동시에 처리하기 위해 모든 상태 파일을
`listenlist/sessions/<broadcast_id>/` 하위로 격리한다.

broadcast_id가 비어 있으면(예: 단독 마이크 테스트) "default" 세션을 사용한다.
Node 백엔드(chatListenListStore.js)도 동일한 규칙으로 경로를 계산하므로
chat.jsonl을 같은 위치에서 주고받는다.
"""

import os
from pathlib import Path

_BASE = Path(__file__).parent
_SESSIONS_ROOT = _BASE / "sessions"

# 한 세션 디렉터리에 존재하는 파일 목록 (초기화 대상)
SESSION_FILES = (
    "transcriptions.jsonl",
    "chat.jsonl",
    "ai_outputs.jsonl",
    "ready_summary.json",
    "ready_question.json",
)


def safe_id(broadcast_id: str | None) -> str:
    """경로에 안전한 세션 식별자로 정규화한다 (경로 조작 방지)."""
    bid = (broadcast_id or "").strip()
    safe = "".join(c for c in bid if c.isalnum() or c in ("-", "_"))
    return safe or "default"


def resolve_id(broadcast_id: str | None = None) -> str:
    """명시 인자 → BROADCAST_ID 환경변수 → "default" 순으로 세션 ID를 결정한다."""
    return safe_id(broadcast_id if broadcast_id is not None and broadcast_id != "" else os.getenv("BROADCAST_ID", ""))


def session_dir(broadcast_id: str | None = None) -> Path:
    """세션 디렉터리 경로를 반환하고 없으면 생성한다."""
    d = _SESSIONS_ROOT / resolve_id(broadcast_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def transcriptions_path(broadcast_id: str | None = None) -> Path:
    return session_dir(broadcast_id) / "transcriptions.jsonl"


def chat_path(broadcast_id: str | None = None) -> Path:
    return session_dir(broadcast_id) / "chat.jsonl"


def ai_outputs_path(broadcast_id: str | None = None) -> Path:
    return session_dir(broadcast_id) / "ai_outputs.jsonl"


def ready_summary_path(broadcast_id: str | None = None) -> Path:
    return session_dir(broadcast_id) / "ready_summary.json"


def ready_question_path(broadcast_id: str | None = None) -> Path:
    return session_dir(broadcast_id) / "ready_question.json"
