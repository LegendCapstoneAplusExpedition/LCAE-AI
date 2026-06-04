"""High-confidence broadcast responses that must stay stable across LLM runs."""

from collections.abc import Iterable


MVP_SUMMARY = (
    "오늘은 첫 제품 만들기를 이야기 나눠봤네요. MVP로 핵심 기능 하나에 집중하고, "
    "반응 속도까지 UX로 챙기며, 사용자 피드백을 작은 단위 개선으로 잇는 게 핵심이었네요."
)

MVP_CLOSING = (
    "네, 오늘도 함께해 주셔서 감사합니다. MVP, 속도, 피드백까지 첫 제품을 단단하게 만드는 "
    "핵심을 짚어봤어요. 다음 멘토링에서 또 뵙겠습니다."
)

def opening_script(topics: Iterable[str]) -> str:
    """AI 진행자가 방송 합류 직후 1회 발화하는 오프닝 멘트.

    멘토가 사전에 입력한 방송 주제(broadcast_topics)를 바탕으로 인사를 건넨다.
    주제가 없으면 일반 인사로 대체한다.
    """
    items = [str(topic).strip() for topic in topics if str(topic).strip()]
    if not items:
        return "네, 안녕하세요. 오늘도 멘토링 방송 함께 시작해볼게요."
    topic_phrase = ", ".join(items[:3])
    return (
        f"네, 안녕하세요. 오늘은 '{topic_phrase}' 이야기로 "
        "멘토링을 함께 나눠볼 텐데요. 편하게 시작해볼게요."
    )


def _contains_all(text: str, keywords: tuple[str, ...]) -> bool:
    return all(keyword.casefold() in text.casefold() for keyword in keywords)


def curated_bridge(mentor_text: str) -> str:
    """Return a stable bridge for a clearly recognizable key moment."""
    text = (mentor_text or "").strip()
    rules = [
        (
            (("첫", "제품", "만들"), ("첫", "사이드", "프로젝트")),
            "첫 제품 만들기, 많은 분들이 궁금해하는 주제네요.",
        ),
        (
            (("MVP", "핵심"), ("MVP", "검증")),
            "핵심 기능 하나에 집중하자는 얘기네요.",
        ),
        (
            (("반응", "늦"), ("속도", "UX")),
            "결국 속도도 UX의 일부라는 말이네요.",
        ),
        (
            (("피드백", "개선"),),
            "피드백을 다음 개선으로 잇는 흐름이네요.",
        ),
        (
            (("커뮤니케이션", "공유"), ("역할", "커뮤니케이션")),
            "공유 방식이 성패를 가르는 지점이네요.",
        ),
        (
            (("사용자", "흐름"),),
            "사용자 흐름을 기준으로 보자는 얘기네요.",
        ),
    ]
    for alternatives, bridge in rules:
        if any(_contains_all(text, keywords) for keywords in alternatives):
            return bridge
    return ""


def curated_summary(transcripts: Iterable[str]) -> str:
    """Return the canonical summary when the MVP broadcast beats are present."""
    joined = "\n".join(str(text) for text in transcripts)
    has_mvp = "MVP".casefold() in joined.casefold()
    has_core = _contains_all(joined, ("핵심", "기능"))
    has_speed = any(
        keyword.casefold() in joined.casefold() for keyword in ("UX", "사용자 경험")
    ) and any(
        keyword in joined
        for keyword in ("반응", "속도")
    )
    has_feedback = _contains_all(joined, ("피드백", "개선"))

    if has_mvp and has_core and has_speed and has_feedback:
        return MVP_SUMMARY
    return ""


def curated_closing(transcripts: Iterable[str]) -> str:
    """Return the canonical closing for the MVP broadcast scenario."""
    if curated_summary(transcripts):
        return MVP_CLOSING
    return ""
