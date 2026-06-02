"""
ListenList — ASR 전사 결과를 JSONL 파일로 관리하는 버퍼.

각 항목 형식:
    {"time": "2026-05-29 14:23:05", "text": "발화 내용", "conf": 0.4}

규칙:
    - append()로 항목 추가, SUMMARY_INTERVAL개마다 백그라운드에서 요약 갱신.
    - 요약 소스는 실제 전사 텍스트 (LLM 생성 내용 아님) → 할루시네이션 없음.
    - 요약 결과는 ready_summary.json에 덮어씀 (정리요청 시 즉시 읽힘).
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

_DEFAULT_PATH    = Path(__file__).parent / "transcriptions.jsonl"
_READY_SUMMARY_PATH = Path(__file__).parent / "ready_summary.json"

SUMMARY_INTERVAL = 3  # 전사 N개마다 백그라운드 요약 갱신


class ListenList:
    def __init__(self, path: Path = _DEFAULT_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._summarizing = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, text: str, conf: float) -> dict:
        """전사 결과를 JSONL에 추가. SUMMARY_INTERVAL개마다 백그라운드 요약 트리거."""
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "conf": round(conf, 4),
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            entries = self._read()
            should_summarize = (
                len(entries) % SUMMARY_INTERVAL == 0
                and len(entries) > 0
                and not self._summarizing
            )
            if should_summarize:
                self._summarizing = True
                snapshot = list(entries)

        if should_summarize:
            threading.Thread(
                target=self._summarize_to_ready, args=(snapshot,), daemon=True
            ).start()

        return entry

    def read_all(self) -> list[dict]:
        """transcriptions.jsonl의 모든 항목을 반환."""
        with self._lock:
            return self._read()

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────────

    def _summarize_to_ready(self, entries: list[dict]) -> None:
        """실제 전사 텍스트 기반으로 LLM 요약 → ready_summary.json 저장."""
        try:
            from langchain_core.messages import HumanMessage
            from pipeline.llm.utils.llm import llm

            texts = "\n".join(f"[{e['time']}] {e['text']}" for e in entries)
            prompt = (
                "다음은 방송에서 실제로 전사된 내용입니다.\n\n"
                f"{texts}\n\n"
                "위 내용만을 바탕으로 3문장 이내로 요약하세요. "
                "전사된 내용 외의 정보나 추측은 절대 추가하지 마세요. "
                "다른 설명 없이 요약문만 출력하세요."
            )
            result = llm.invoke([HumanMessage(content=prompt)])
            summary = result.content.strip()

            _READY_SUMMARY_PATH.write_text(
                json.dumps(
                    {"time": time.strftime("%H:%M:%S"), "summary": summary},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"[ListenList] 요약 갱신 완료: {summary[:60]}...")
        except Exception as e:
            print(f"[ListenList] 요약 실패: {e}")
        finally:
            with self._lock:
                self._summarizing = False

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
