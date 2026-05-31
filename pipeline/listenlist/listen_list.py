"""
ListenList — ASR 전사 결과를 JSONL 파일로 관리하는 버퍼.

각 항목 형식:
    {"time": "2026-05-29 14:23:05", "text": "발화 내용", "conf": 0.4}

규칙:
    - append()로 항목 추가, BUFFER_SIZE(30)개 누적 시 LLM 요약 자동 트리거.
    - 요약 완료 후 처리된 30개 항목은 transcriptions.jsonl에서 제거.
    - 요약 결과는 summary.jsonl에 {time, keywords, summary} 형태로 축적.
"""

import json
import threading
from datetime import datetime
from pathlib import Path

_DEFAULT_PATH = Path(__file__).parent / "transcriptions.jsonl"
_SUMMARY_PATH = Path(__file__).parent / "summary.jsonl"
BUFFER_SIZE = 10


class ListenList:
    def __init__(
        self,
        path: Path = _DEFAULT_PATH,
        summary_path: Path = _SUMMARY_PATH,
        buffer_size: int = BUFFER_SIZE,
    ):
        self.path = Path(path)
        self.summary_path = Path(summary_path)
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        self._summarizing = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, text: str, conf: float) -> dict:
        """전사 결과를 JSONL에 추가. buffer_size 도달 시 비동기 요약 트리거."""
        entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "conf": round(conf, 4),
        }
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            entries = self._read()
            should_summarize = len(entries) >= self.buffer_size and not self._summarizing
            if should_summarize:
                self._summarizing = True
                batch = entries[: self.buffer_size]

        if should_summarize:
            threading.Thread(
                target=self._summarize_and_clear, args=(batch,), daemon=True
            ).start()

        return entry

    def read_all(self) -> list[dict]:
        """transcriptions.jsonl의 모든 항목을 반환."""
        with self._lock:
            return self._read()

    def read_summaries(self) -> list[dict]:
        """summary.jsonl의 모든 요약을 반환."""
        with self._lock:
            return self._read_file(self.summary_path)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _summarize_and_clear(self, batch: list[dict]) -> None:
        """LLM으로 batch 요약 → summary.jsonl 저장 → transcriptions.jsonl 앞 N줄 제거."""
        try:
            summary_entry = self._call_llm(batch)
            with self._lock:
                with open(self.summary_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
                current = self._read()
                self._write(current[len(batch):])
            print(f"[ListenList] 요약 완료: {summary_entry['time']} / 키워드: {summary_entry['keywords']}")
        except Exception as e:
            print(f"[ListenList] 요약 실패: {e}")
        finally:
            with self._lock:
                self._summarizing = False

    def _call_llm(self, batch: list[dict]) -> dict:
        """batch 항목들을 LLM으로 요약하여 {time, keywords, summary} 반환."""
        from langchain_core.prompts import ChatPromptTemplate
        from pipeline.llm.utils.llm import llm_structured

        texts = "\n".join(f"[{e['time']}] {e['text']}" for e in batch)
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        prompt = ChatPromptTemplate.from_template(
            "다음은 실시간 음성 인식으로 수집된 대화 내용입니다.\n\n"
            "{texts}\n\n"
            "위 내용을 분석하여 JSON으로만 응답하세요 (다른 텍스트 없이):\n"
            '{{"keywords": ["핵심키워드1", "핵심키워드2", "핵심키워드3"], '
            '"summary": "대화의 흐름과 중요 내용을 2~3문장으로 요약"}}'
        )

        chain = prompt | llm_structured
        result = chain.invoke({"texts": texts})

        content = result.content if hasattr(result, "content") else str(result)
        parsed = json.loads(content) if isinstance(content, str) else content

        return {
            "time": time_str,
            "keywords": parsed.get("keywords", []),
            "summary": parsed.get("summary", ""),
        }

    def _read(self) -> list[dict]:
        return self._read_file(self.path)

    def _read_file(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        result = []
        for line in path.read_text(encoding="utf-8").splitlines():
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
