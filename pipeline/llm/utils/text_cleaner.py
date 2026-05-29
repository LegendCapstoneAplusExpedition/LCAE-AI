"""
한국어 구어체 필러 제거 유틸리티 (regex 기반)

LLM 전처리 대비 속도: ~0ms vs ~12s
커버 범위: 알려진 패턴만 처리. 미등록 표현은 통과.
재학습 후: preprocess_node를 다시 LLM 방식으로 교체 가능.
"""

import re

# 문장 첫머리 단독 추임새: "아 ", "어어 ", "음 " 등
_LEADING_INTERJECTION = re.compile(
    r'^(?:아+|어+|음+|으+|에+|예+)\s+'
)

# 구어체 전환·메타 표현 (어느 위치든)
_FILLER_PHRASES = re.compile(
    r'(?:'
    r'그니까|그러니까'         # 그니까 / 그러니까
    r'|제\s*말은'             # 제 말은
    r'|뭐랄까'               # 뭐랄까
    r'|있잖아요?'            # 있잖아 / 있잖아요
    r'|어떻게\s*말하면'       # 어떻게 말하면
    r'|그게\s*뭐냐면'         # 그게 뭐냐면
    r'|솔직히\s*말하면'       # 솔직히 말하면
    r')\s*[,，]?\s*'
)

# 공백 정리
_MULTI_SPACE = re.compile(r'\s+')


def clean_fillers(text: str) -> str:
    """
    구어체 필러를 제거하고 정제된 텍스트를 반환한다.
    원문 의미에 영향을 주지 않는 표현만 제거한다.
    """
    text = _LEADING_INTERJECTION.sub('', text)
    text = _FILLER_PHRASES.sub('', text)
    text = _MULTI_SPACE.sub(' ', text).strip()
    return text
