"""
ListenList — ASR 전사 결과를 JSONL 파일로 관리하는 버퍼.

각 항목 형식:
    {"time": "2026-05-29 14:23:05", "text": "발화 내용", "conf": 0.4}

규칙:
    - append()로 항목 추가, SUMMARY_INTERVAL개마다 백그라운드에서 요약 갱신.
    - 요약 소스는 실제 전사 텍스트 (LLM 생성 내용 아님) → 할루시네이션 없음.
    - 이전 요약 + 새 전사 tail을 요약하며, 최근 전사에 더 큰 비중을 둠.
    - 요약 결과는 ready_summary.json에 덮어씀 (정리요청 시 즉시 읽힘).

파일 경로는 broadcast_id로 세션 격리된다 (동시 방송 중첩 방지):
    sessions/<broadcast_id>/transcriptions.jsonl  ← 전사 누적
    sessions/<broadcast_id>/ready_summary.json    ← 요약 결과
broadcast_id가 없으면 "default" 세션을 사용한다. (paths.py 참조)
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

from pipeline.listenlist.paths import transcriptions_path, ready_summary_path

SUMMARY_INTERVAL = 3  # 전사 N개마다 백그라운드 요약 갱신


class ListenList:
    def __init__(self, broadcast_id: str | None = None, path: Path | None = None):
        # broadcast_id로 세션별 파일 경로를 결정한다 (동시 방송 격리).
        self.broadcast_id = broadcast_id
        self.path = Path(path) if path is not None else transcriptions_path(broadcast_id)
        self._ready_summary_path = ready_summary_path(broadcast_id)
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

    def _summarize_to_ready(self, entries: list[dict]) -> dict | None:
        """이전 요약 + 새 전사 tail 기반으로 LLM 요약 → ready_summary.json 저장."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from pipeline.llm.utils.llm import llm_summary, llm_lock

            previous = self._read_ready_summary()
            previous_summary = str(previous.get("summary", "")).strip()
            try:
                source_count = int(previous.get("source_count", 0) or 0)
            except (TypeError, ValueError):
                source_count = 0
            if source_count < 0 or source_count > len(entries):
                previous_summary = ""
                source_count = 0

            new_entries = entries[source_count:]
            if not new_entries and previous_summary:
                print("[ListenList] 새 요약 대상 없음 → 기존 요약 유지")
                return previous

            recent_entries = new_entries[-SUMMARY_INTERVAL:]
            new_texts = "\n".join(f"[{e['time']}] {e['text']}" for e in new_entries)
            recent_texts = "\n".join(f"[{e['time']}] {e['text']}" for e in recent_entries)

            if previous_summary:
                prompt = (
                    "다음은 방송의 이전 누적 요약과 새로 추가된 실제 멘토 전사입니다.\n\n"
                    f"[이전 누적 요약]\n{previous_summary}\n\n"
                    f"[새로 추가된 전사]\n{new_texts}\n\n"
                    f"[최근 전사 - 더 높은 비중]\n{recent_texts}\n\n"
                    "요약 갱신 규칙:\n"
                    "- 이전 누적 요약과 새 전사에 명시된 내용만 바탕으로 누적 요약을 갱신하세요.\n"
                    "- 최근 전사에 나온 새 주제, 방향 전환, 결론, 강조점을 이전 내용보다 더 크게 반영하세요.\n"
                    "- 오래된 내용은 핵심 맥락만 유지하고, 최근 내용과 중복되면 압축하세요.\n"
                    "- 전사에 없는 주제, 배경, 조언, 학습 데이터의 표현을 추가하지 마세요.\n"
                    "- 잡음, 인사, 테스트 발화, 요약 요청 문장 자체는 핵심 내용이 아니면 제외하세요.\n"
                    "- 새 전사에 핵심 내용이 없으면 이전 누적 요약을 그대로 출력하세요.\n"
                    "- 1~3문장으로, 다른 설명 없이 요약문만 출력하세요."
                )
            else:
                prompt = (
                    "다음은 방송에서 실제로 전사된 멘토 발화 목록입니다.\n\n"
                    f"[전사]\n{new_texts}\n\n"
                    f"[최근 전사 - 더 높은 비중]\n{recent_texts}\n\n"
                    "요약 규칙:\n"
                    "- 위 전사에 명시된 내용만 바탕으로 요약하세요.\n"
                    "- 최근 전사에 나온 새 주제, 방향 전환, 결론, 강조점을 더 크게 반영하세요.\n"
                    "- 전사에 없는 주제, 배경, 조언, 학습 데이터의 표현을 추가하지 마세요.\n"
                    "- 잡음, 인사, 테스트 발화, 요약 요청 문장 자체는 핵심 내용이 아니면 제외하세요.\n"
                    "- 핵심 방송 내용이 부족하면 정확히 '아직 요약할 핵심 내용이 없습니다.'라고만 출력하세요.\n"
                    "- 1~3문장으로, 다른 설명 없이 요약문만 출력하세요."
                )

            # 요약 전용 system 프롬프트. 메인 모델(driving-mentor)을 재사용하는 경우
            # Modelfile의 멘토 SYSTEM(구조화 JSON 강제)을 이 메시지로 덮어써서
            # 평문 요약을 받는다. Ollama는 요청에 system 메시지가 있으면 그것을 우선한다.
            system_prompt = (
                "당신은 방송 전사 요약기입니다. 입력된 전사 내용만 바탕으로 한국어 평문 "
                "요약문을 작성하세요. JSON·키-값·대괄호 태그·코드블록·머리말 없이, "
                "요약 문장 자체만 출력합니다."
            )

            # 메인 파이프라인(analyze_write)과 같은 Ollama 모델 슬롯을 동시에 치지
            # 않도록 직렬화한다. 요약은 백그라운드 작업이므로 메인 호출이 진행 중이면
            # 락을 기다렸다가 그 뒤에 실행된다(동시 호출로 인한 정지 방지).
            with llm_lock:
                result = llm_summary.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt),
                ])
            summary = result.content.strip()

            ready_summary = {
                "time": time.strftime("%H:%M:%S"),
                "summary": summary,
                "source_count": len(entries),
                "previous_source_count": source_count,
                "new_count": len(new_entries),
            }
            self._ready_summary_path.write_text(
                json.dumps(ready_summary, ensure_ascii=False),
                encoding="utf-8",
            )
            print(
                f"[ListenList] 요약 갱신 완료: "
                f"{source_count}→{len(entries)} ({len(new_entries)}개 tail): {summary[:60]}..."
            )
            return ready_summary
        except Exception as e:
            print(f"[ListenList] 요약 실패: {e}")
            return None
        finally:
            with self._lock:
                self._summarizing = False

    def _read_ready_summary(self) -> dict:
        if not self._ready_summary_path.exists():
            return {}
        try:
            data = json.loads(self._ready_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

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
