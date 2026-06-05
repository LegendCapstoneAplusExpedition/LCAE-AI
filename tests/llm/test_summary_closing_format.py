import json
import unittest
from uuid import uuid4

from pipeline.listenlist.paths import ready_summary_path, session_dir
from pipeline.llm.chain.nodes import output_node, summarize_listenlist_node


class SummaryClosingFormatTest(unittest.TestCase):
    def setUp(self):
        self.broadcast_id = f"test-{uuid4().hex}"

    def tearDown(self):
        directory = session_dir(self.broadcast_id)
        for path in directory.glob("*"):
            path.unlink()
        directory.rmdir()

    def test_summary_request_starts_with_required_topic_sentence(self):
        ready_summary_path(self.broadcast_id).write_text(
            json.dumps(
                {
                    "summary": "MVP는 핵심 기능 하나를 빠르게 검증하는 방식입니다. 사용자 반응을 보며 작은 단위로 개선합니다."
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        state = {
            "broadcast_id": self.broadcast_id,
            "broadcast_topics": ["사이드 프로젝트 MVP 만들기"],
            "current_topic": "사이드 프로젝트 MVP",
        }

        result = summarize_listenlist_node(state)

        self.assertEqual(
            result["mc_script"],
            "오늘은 사이드 프로젝트 MVP에 대해 이야기했습니다. MVP는 핵심 기능 하나를 빠르게 검증하는 방식입니다. 사용자 반응을 보며 작은 단위로 개선합니다.",
        )

    def test_closing_uses_required_fixed_format(self):
        state = {
            "broadcast_id": self.broadcast_id,
            "broadcast_topics": ["사이드 프로젝트 MVP"],
            "current_topic": "사이드 프로젝트 MVP",
            "context_summary": "MVP는 핵심 기능 하나를 빠르게 검증하는 방식입니다. 사용자 반응을 보며 작은 단위로 개선합니다.",
            "intent": "마무리",
            "mc_script": "오늘 방송은 여기까지 하겠습니다.",
        }

        result = output_node(state)

        self.assertEqual(
            result["messages"][0].content,
            "네, 오늘도 함께해주셔서 감사합니다. MVP는 핵심 기능 하나를 빠르게 검증하는 방식이라는 점을 짚어봤습니다. 다음 멘토링에서 또 뵙겠습니다.",
        )

    def test_summary_topic_and_malformed_ending_are_normalized(self):
        ready_summary_path(self.broadcast_id).write_text(
            json.dumps(
                {
                    "summary": "작업 속도는 완벽한 결과를 한 번에 만들려고 할 때 느려집니다. 작은 결과물을 먼저 완성하고 점진적으로 개선하는 게 좋아요. 집중력을 유지하려면 짧은 브레이크를 취하면 도움이요"
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        state = {
            "broadcast_id": self.broadcast_id,
            "broadcast_topics": ["개발자가 업무에 집중을 유지하는 방법"],
            "current_topic": "작업 범위를 좁혀야 집중이 가능합니다",
        }

        result = summarize_listenlist_node(state)

        self.assertEqual(
            result["mc_script"],
            "오늘은 작업 범위 집중에 대해 이야기했습니다. 작업 속도는 완벽한 결과를 한 번에 만들려고 할 때 느려집니다. 작은 결과물을 먼저 완성하고 점진적으로 개선하는 게 좋아요. 집중력을 유지하려면 짧은 브레이크를 취하면 도움이 됩니다.",
        )

    def test_closing_normalizes_jibnida_ending(self):
        state = {
            "broadcast_id": self.broadcast_id,
            "broadcast_topics": ["개발자가 업무에 집중을 유지하는 방법"],
            "current_topic": "작업 범위를 좁혀야 집중이 가능합니다",
            "context_summary": "작업 속도는 완벽한 결과를 한 번에 만들려고 할 때 느려집니다. 작은 결과물을 먼저 완성하고 점진적으로 개선하는 게 좋아요.",
            "intent": "마무리",
            "mc_script": "",
        }

        result = output_node(state)

        self.assertEqual(
            result["messages"][0].content,
            "네, 오늘도 함께해주셔서 감사합니다. 작업 속도는 완벽한 결과를 한 번에 만들려고 할 때 느려진다는 점을 짚어봤습니다. 다음 멘토링에서 또 뵙겠습니다.",
        )


if __name__ == "__main__":
    unittest.main()
