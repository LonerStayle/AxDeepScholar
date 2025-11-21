
from datetime import datetime
from langchain.chat_models import init_chat_model
from main_state import MainInputState, MainState, ClarifyWithUser, ResearchQuestion
from langgraph.graph.state import Command, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from utils.helper import get_today_str
from core.prompts.PromptManager import PromptManager
from core.prompts.PromptType import PromptType

from agent.research.research_agent import graph as research_graph
from agent.supervisor.supervisor_agent import graph as supervisor_graph

prompt_manager = PromptManager()
model = init_chat_model(model="gpt-4.1", temperature=0)
final_model = init_chat_model(model="gpt-5.1", model_provider="openai")


def clarify_with_user(
    state: MainState,
) -> Command[Literal["write_research_brief", "__end__"]]:
    clarify_prompt = prompt_manager.get_prompt(
        PromptType.SYSTEM_CHECK_USER_NEED,
        messages=state["messages"],
        date=get_today_str(),
    )
    parser_model = model.with_structured_output(ClarifyWithUser)
    response: ClarifyWithUser = parser_model.invoke(
        [HumanMessage(content=clarify_prompt)]
    )

    if response.need_clarification:
        return Command(
            goto=END, update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]},
        )


def write_research_brief(state: MainState) -> MainState:
    parser_model = model.with_structured_output(ResearchQuestion)
    start_prompt = prompt_manager.get_prompt(
        PromptType.START_RESEARCH, messages=state["messages"], date=get_today_str()
    )
    response: ResearchQuestion = parser_model.invoke(
        [HumanMessage(content=start_prompt)]
    )
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=response.research_brief)],
    }


async def final_report(state: MainState) -> MainState:
    notes = state.get("notes", [])
    findings = "\n".join(notes)
    final_report_prompt = prompt_manager.get_prompt(
        PromptType.FINAL_PEPORT,
        research_brief=state.get("research_brief", ""),
        date=get_today_str(),
        findings=findings,
    )
    final_report = await final_model.ainvoke(
        [HumanMessage(content=final_report_prompt)]
    )
    return {
        "final_report": final_report.content,
        "messages": ["이것이 최종 리포트 입니다." + final_report.content],
    }


graph_builder = StateGraph(MainState, input_schema=MainInputState)
graph_builder.add_node("clarify_with_user", clarify_with_user)
graph_builder.add_node("write_research_brief", write_research_brief)
graph_builder.add_node("supervisor_graph",supervisor_graph)
graph_builder.add_node("final_report",final_report)


graph_builder.add_edge(START, "clarify_with_user")
graph_builder.add_edge("write_research_brief", "supervisor_graph")
graph_builder.add_edge("supervisor_graph", "final_report")
graph_builder.add_edge("final_report",END)

graph = graph_builder.compile()
