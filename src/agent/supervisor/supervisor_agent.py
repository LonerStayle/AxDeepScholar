from langchain_core.messages import (
    BaseMessage,
    filter_messages,
    SystemMessage,
    ToolMessage,
)
from langchain.chat_models import init_chat_model
from tools import think_tool
from agent.supervisor.supervisor_state import (
    SupervisorState,
    ConductResearch,
    ResearchComplete,
)
from langgraph.types import Command
from typing_extensions import Literal
from utils.helper import get_today_str
from core.prompts.PromptManager import PromptManager
from core.prompts.PromptType import PromptType


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """감독자(supervisor)의 메시지 기록에서 ToolMessage 객체에 포함된 연구 노트를 추출합니다.

    이 함수는 하위 에이전트(sub-agent)가 반환한 압축된 연구 결과를 가져옵니다.

    감독자가 ConductReasearch 툴 호출을 통해 연구를 하위 에이전트에게 위임하면,
    각 하위 에이전트는 자신이 요약한 연구 결과를 ToolMessage의 content로 반환합니다.

    이 함수는 그러한 ToolMessage의 내용을 모두 추출하여,
    최종 연구 노트 리스트로 정리합니다.
    """

    return [
        tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")
    ]


supervisor_tools = [ConductResearch, ResearchComplete, think_tool]

try:
    import nest_asyncio

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass
except ImportError:
    pass


supervisor_tools = [ConductResearch, ResearchComplete, think_tool]
supervisor_model = init_chat_model(model="gpt-5.1", model_provider="openai")

supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

max_researcher_iterations = 5
max_concurrent_researchers = 3

prompt_manager = PromptManager()


async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    supervisor_prompt = prompt_manager.get_prompt(
        PromptType.SYSTEM_SUPERVISOR,
        date=get_today_str(),
        max_researcher_iterations=max_researcher_iterations,
        max_concurrent_research_units=max_concurrent_researchers,
    )

    system_message = SystemMessage(content=supervisor_prompt)
    messages = [system_message] + state.get("supervisor_messages")
    response = await supervisor_model_with_tools.ainvoke(messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


from agent.research.research_agent import graph as research_graph
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
from typing_extensions import Literal
import asyncio


async def supervisor_tools(
    state: SupervisorState,
) -> Command[Literal["supervisor", "__end__"]]:
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)

    last_message = supervisor_messages[-1]

    tool_messages = []
    all_raw_notes = []
    next_step = "supervisor"
    should_end = False

    exit_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not last_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete" for tool_call in last_message.tool_calls
    )

    if exit_iterations or no_tool_calls or research_complete:
        should_end = True
        next_step = END
    else:
        try:
            think_tool_calls = [
                tool_call
                for tool_call in last_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call
                for tool_call in last_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            for tool_call in think_tool_calls:
                observation = think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

            if conduct_research_calls:
                coros = [
                    research_graph.ainvoke(
                        {
                            "researcher_messages": [
                                HumanMessage(
                                    content=tool_call["args"]["research_topic"]
                                )
                            ],
                            "research_topic": tool_call["args"]["research_topic"],
                        }
                    )
                    for tool_call in conduct_research_calls
                ]

                conduct_research_calls_results = await asyncio.gather(*coros)

                research_tool_messages = [
                    ToolMessage(
                        content=result.get(
                            "compressed_research", "연구 보고서를 합성하는 중 오류 발생"
                        ),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    for result, tool_call in zip(
                        conduct_research_calls_results, conduct_research_calls
                    )
                ]
                tool_messages.extend(research_tool_messages)
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", []))
                    for result in conduct_research_calls_results
                ]
        except Exception as e:
            print(f"Error in supervisor tools: {e}")
            should_end = True
            next_step = END

        if should_end:
            return Command(
                goto=next_step,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", ""),
                },
            )
        else:
            return Command(
                goto=next_step,
                update={
                    "supervisor_messages": tool_messages,
                    "raw_notes": all_raw_notes,
                },
            )


graph_builder = StateGraph(SupervisorState)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("supervisor_tools", supervisor_tools)
graph_builder.add_edge(START, "supervisor")
graph = graph_builder.compile()
