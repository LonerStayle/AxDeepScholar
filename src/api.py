
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from agent.main.main_agent import graph as main_graph
from agent.main.main_state import MainInputState
from langchain_core.messages import HumanMessage, AIMessage


app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

def convert_messages(messages):
    """user → HumanMessage, assistant → AIMessage 변환"""
    converted = []
    for m in messages:
        converted.append(HumanMessage(content=m.content))
    return converted

@app.post("/run")
async def run_agent(req: ChatRequest):
    lc_messages = convert_messages(req.messages)
    state = MainInputState(messages=lc_messages)
    result = await main_graph.ainvoke(state)

    return {
        "messages": result.get("messages"),
        "final_report": result.get("final_report"),
        "research_brief": result.get("research_brief"),
        "notes": result.get("notes")
    }

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
