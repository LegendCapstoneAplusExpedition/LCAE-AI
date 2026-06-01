"""
ChatList — 실시간 채팅을 JSONL 파일로 관리하고 질문을 큐처럼 소비합니다.

각 항목 형식:
    {
        "time": "2026-06-01T12:00:00.000Z",
        "broadcast_id": "...",
        "user_id": "...",
        "username": "devmentor",
        "message": "이직 준비는 언제 시작하면 좋나요?",
        "is_question": true,
        "used": false
    }
"""

import json
import re
import threading
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent / "chat.jsonl"

_QUESTION_RE = re.compile(
    r"(\?|？|질문|궁금|어떻게|왜|뭐|무엇|언제|어디|누구|가능한가|될까요|인가요|나요|까요|알려주세요)"
)


class ChatList:
    def __init__(self, path: Path = _DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def read_all(self) -> list[dict]:
        with self._lock:
            return self._read()

    def pop_next_question(self, broadcast_id: str | None = None) -> dict | None:
        """아직 사용하지 않은 첫 질문 채팅을 반환하고 used=true로 표시."""
        with self._lock:
            entries = self._read()
            for entry in entries:
                if entry.get("used"):
                    continue
                if broadcast_id and entry.get("broadcast_id") != broadcast_id:
                    continue
                message = str(entry.get("message", "")).strip()
                if entry.get("is_question") or _QUESTION_RE.search(message):
                    entry["is_question"] = True
                    entry["used"] = True
                    self._write(entries)
                    return entry
        return None

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []

        result = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return result

    def _write(self, entries: list[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
