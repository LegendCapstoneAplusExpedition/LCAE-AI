import unittest

from pipeline.llm.chain.scripted_responses import (
    MVP_CLOSING,
    MVP_SUMMARY,
    curated_bridge,
    curated_closing,
    curated_summary,
)


class ScriptedResponsesTest(unittest.TestCase):
    def setUp(self):
        self.transcripts = [
            "안녕하세요, 오늘은 첫 사이드 프로젝트를 진짜 제품으로 만드는 법을 이야기해볼게요.",
            "처음부터 다 만들려고 하지 마세요. MVP는 핵심 기능 하나를 빠르게 검증하는 겁니다.",
            "그리고 버튼을 눌렀을 때 반응이 늦으면 사용자 경험이 뚝 끊깁니다.",
            "좋은 질문이에요. 사용자 피드백을 받으면 작은 단위로 바로 개선하세요.",
        ]

    def test_example_bridges_are_stable(self):
        expected = [
            "첫 제품 만들기, 많은 분들이 궁금해하는 주제네요.",
            "핵심 기능 하나에 집중하자는 얘기네요.",
            "결국 속도도 UX의 일부라는 말이네요.",
            "피드백을 다음 개선으로 잇는 흐름이네요.",
        ]

        self.assertEqual(
            [curated_bridge(text) for text in self.transcripts],
            expected,
        )

    def test_example_summary_and_closing_are_stable(self):
        self.assertEqual(curated_summary(self.transcripts), MVP_SUMMARY)
        self.assertEqual(curated_closing(self.transcripts), MVP_CLOSING)

    def test_unrecognized_topic_falls_back_to_llm(self):
        self.assertEqual(curated_bridge("오늘은 데이터베이스 인덱스를 설명합니다."), "")


if __name__ == "__main__":
    unittest.main()
