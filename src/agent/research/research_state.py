from typing import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
import operator

class ResearcherState(TypedDict):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]

class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]

class Summary(BaseModel):
    summary: str = Field(description="웹페이지 콘텐츠의 간결한 요약")
    key_excerpts:str = Field(description="콘텐츠에서 발췌한 중요한 인용문과 핵심 구절")

