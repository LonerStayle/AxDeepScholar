from utils.helper import attach_auto_keys
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated, Optional, Sequence
import operator


@attach_auto_keys
class MainInputState(MessagesState):
    pass


@attach_auto_keys
class MainState(MessagesState):
    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add] = []
    notes: Annotated[list[str], operator.add] = []
    fianl_report: str


@attach_auto_keys
class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(
        description="사용자에게 추가 설명을 해야하는지 여부"
    )
    question: str = Field(
        description="논문 내용을 구체적으로 작성하기 위해 사용자에게 제시할 추가 질문 입니다."
    )
    verification: str = Field(
        description="사용자가 필수 정보를 모두 제공해줘서 실제 연구를 시작할 것임을 전달하는 메시지 입니다."
    )


@attach_auto_keys
class ResearchQuestion(BaseModel):
    research_brief: str = Field(
        description="연구의 방향과 범위를 결정하기 위해 생성된 핵심 연구 질문입니다."
    )


