"""
AIOutputList — LLM이 생성한 분석 결과와 최종 AI 발화를 JSONL로 저장합니다.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline.listenlist.paths import ai_outputs_path


class AIOutputList:
    def __init__(self, broadcast_id: str | None = None, path: Path | None = None):
        # broadcast_id로 세션별 파일 경로를 결정한다 (동시 방송 격리).
        self.broadcast_id = broadcast_id
        self.path = Path(path) if path is not None else ai_outputs_path(broadcast_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(
        self,
        *,
        mentor_text: str,
        mentor_confidence: float,
        state: dict[str, Any],
        ai_text: str,
        spoken: bool,
    ) -> dict:
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "broadcast_id": self.broadcast_id or os.getenv("BROADCAST_ID", ""),
            "mentor_text": mentor_text,
            "mentor_confidence": round(float(mentor_confidence), 4),
            "topic": state.get("current_topic", ""),
            "summary": state.get("context_summary", ""),
            "intent": state.get("intent", ""),
            "streaming_stage": state.get("streaming_stage", ""),
            "mc_script": state.get("mc_script", ""),
            "pending_question": state.get("pending_question", ""),
            "ai_text": ai_text,
            "spoken": spoken,
        }

        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []

        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows
