
import operator 
from typing_extensions import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool 
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class SupervisorState(TypedDict):
    supervisor_messages:Annotated[Sequence[BaseMessage],add_messages]
    research_brief:str 
    notes:Annotated[list[str],operator.add] = [] 
    research_iterations:int = 0 
    raw_notes:Annotated[list[str],operator.add] = [] 

@tool
class ConductResearch(BaseModel):
    """이 도구는 연구를 세분화하여, 특정 주제에 더 잘맞는 하위 에이전트에게 연구 작업을 위힘하도록 돕습니다."""
    research_topic:str = Field(
        description="연구할 주제입니다. 하나의 단일 주제여야 하며, 최소한 하는 한 단락이상으로 구체적으로 자세히 설명되어야 합니다."
    )    

@tool
class ResearchComplete(BaseModel):
    """이제 연구가 완료되었으니, 다음 단계(예: 요약, 보고서 생성등)로 넘겨주세요"""
    pass
