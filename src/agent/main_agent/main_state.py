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
    research_plan: Optional[str]
    supervisor_messages:Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[list[str], operator.add] = []
    notes: Annotated[list[str], operator.add] = []
    fianl_report:str

