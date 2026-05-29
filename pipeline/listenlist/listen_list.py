"""
ListenList — ASR 전사 결과를 JSONL 파일로 관리하는 버퍼.

각 항목 형식:
    {"time": "2026-05-29 14:23:05", "text": "발화 내용", "conf": 0.4}

규칙:
    - append()로 항목 추가 시 max_entries 초과분은 오래된 순으로 자동 삭제.
    - remove_entry()로 LLM 처리 완료된 항목을 명시적으로 삭제.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent / "transcriptions.jsonl"
MAX_ENTRIES = 50  # 이 수를 초과하면 오래된 항목 자동 삭제


class ListenList:
    def __init__(self, path: Path = _DEFAULT_PATH, max_entries: int = MAX_ENTRIES):
        self.path = Path(path)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, text: str, conf: float) -> dict:
        """전사 결과를 JSONL에 추가. max_entries 초과 시 오래된 항목 자동 삭제. 추가된 항목 반환."""
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "conf": round(conf, 4),
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._trim()
        return entry

    def read_all(self) -> list[dict]:
        """파일의 모든 항목을 시간 오름차순으로 반환."""
        with self._lock:
            return self._read()

    def remove_entry(self, time_ms: int) -> None:
        """특정 타임스탬프(ms)의 항목 삭제 — LLM 처리 완료 후 호출."""
        with self._lock:
            entries = self._read()
            kept = [e for e in entries if e["time"] != time_ms]
            if len(kept) != len(entries):
                self._write(kept)

    # ── 내부 헬퍼 (락 안에서만 호출) ──────────────────────────────────────────

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return result

    def _write(self, entries: list[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _trim(self) -> None:
        """max_entries 초과 시 오래된 항목 제거 (락 안에서 호출)."""
        entries = self._read()
        if len(entries) > self.max_entries:
            self._write(entries[-self.max_entries:])
