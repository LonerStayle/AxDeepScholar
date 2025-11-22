from typing_extensions import TypedDict, Annotated, Sequence, List,Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import filter_messages
from agent.research.research_state import ResearcherState, ResearcherOutputState
from utils.helper import get_today_str
from tools.think_tool import think_tool
from tools.retriever_tool import axriv_search
from core.prompts.PromptManager import PromptManager
from core.prompts.PromptType import PromptType
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

tools = [think_tool, axriv_search]
tools_by_name = {tool.name: tool for tool in tools}
model = init_chat_model(model="gpt-5.1-2025-11-13", model_provider="openai")
model_with_tools = model.bind_tools(tools)
summarization_model = init_chat_model(model="gpt-4.1-mini")
compress_model = init_chat_model(model="gpt-4.1")

prompt_manager = PromptManager()


def llm_call(state: ResearcherState) -> ResearcherState:
    system_prompt = prompt_manager.get_prompt(
        PromptType.SYSTEM_RESEARCH, date=get_today_str()
    )
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=system_prompt)] + state["researcher_messages"]
            )
        ]
    }


def tool_node(state: ResearcherState) -> ResearcherState:
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    tool_outputs = [
        ToolMessage(
            content=observation, name=tool_call["name"], tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


def compress_research(state: ResearcherState) -> ResearcherOutputState:
    system_prompt = prompt_manager.get_prompt(
        PromptType.SYSTEM_RESEARCH_COMPRESS_SYSTEM, date=get_today_str()
    )
    human_prompt = prompt_manager.get_prompt(
        PromptType.SYSTEM_RESEARCH_COMPRESS_HUMAN,
        research_topic=state["research_topic"],
    )
    messages = (
        [SystemMessage(content=system_prompt)]
        + state.get("researcher_messages", [])
        + [HumanMessage(content=human_prompt)]
    )
    response = compress_model.invoke(messages)

    filter_mgs = filter_messages(state["researcher_messages"], include_type=["tool", "ai"])
    raw_notes = [str(m.content) for m in filter_mgs]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
    }


def should_continue(
    state: ResearcherState,
) -> Literal["tool_node", "compress_research"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tool_node"
    return "compress_research"


graph_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)
graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("compress_research", compress_research)

graph_builder.add_edge(START, "llm_call")
graph_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {"tool_node": "tool_node", "compress_research": "compress_research"},
)
graph_builder.add_edge("tool_node", "llm_call")
graph_builder.add_edge("compress_research", END)

graph = graph_builder.compile()
