from langchain_core.tools import tool

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """
    연구 과정에서 전략적 판단과 성찰(Reflection)을 기록하는 도구입니다.

    이 도구는 각 검색(Search) 또는 요약(Summarization) 단계 이후,
    현재 상태를 평가하고 다음 단계를 더 정확하게 계획하기 위해 사용됩니다.

    사용 시점 예:
    - 검색 결과를 검토할 때: 이번 검색이 유효했는가?
    - 다음 검색 전략을 세울 때: 추가 정보가 필요한가?
    - 요약을 평가할 때: 논리적 오류나 정보 부족이 있는가?
    - 연구를 마무리하기 전: 이제 충분한 답변을 만들 수 있는가?

    Reflection 내용에는 다음이 포함될 수 있습니다:
    1. 발견한 핵심 정보
    2. 아직 부족한 정보 또는 공백
    3. 현재 전략의 문제점 또는 개선점
    4. 다음 단계에 대한 의사결정(계속 검색 / 요약 생성 / 답변 생성 등)

    Args:
        reflection: 연구 상태에 대한 상세한 성찰 내용

    Returns:
        Reflection이 성공적으로 기록되었음을 의미하는 문자열
    """
    return f"Reflection recorded: {reflection}"
