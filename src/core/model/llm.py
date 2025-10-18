from enum import StrEnum
from langchain_openai import ChatOpenAI
class ModelName(StrEnum):
    GPT_5 = "gpt-5"
    GPT_5_THINKING = "gpt-5-thinking"
    GPT_4_1_MINI = "gpt-4.1-mini"

gpt_5_llm = ChatOpenAI(model = ModelName.GPT_5, temperature=0)
gpt_5_thinking_llm = ChatOpenAI(model = ModelName.GPT_5_THINKING, reasoning_effort="high", verbosity="high")
gpt_4_1_mini_llm = ChatOpenAI(model = ModelName.GPT_4_1_MINI,temperature=0)
