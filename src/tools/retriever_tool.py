from langchain_core.tools import tool
from core.rag.AxrivRetriever import AxrivRetriever

axriv_retriever = AxrivRetriever()

@tool(parse_docstring=True)
def axriv_search(query: str) -> str:
    """
    2024년 10월 ~ 2025년 최신 AI 에이전트 관련 문서를 검색하는 도구입니다.
    이 도구는 LLM의 지식 커트 이외에 최신 정보들을 제공합니다.    

    Args:
        query: 검색 질의

    Returns:
        검색된 문서들의 텍스트
    """
    results = axriv_retriever.hybrid_search(query)

    # Document 객체에서 page_content만 꺼내서 LLM에게 반환
    merged_text = "\n\n---\n\n".join([r.page_content for r in results])
    return merged_text