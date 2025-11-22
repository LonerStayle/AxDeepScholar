from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# 상태, 행동, 보상 저장 예정
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())

# 긍정적 경험 (보상1.0)
vectorstore.add_texts(
    ["State: 비밀번호 입력창 발견. Acction: 'admin123' 입력."],
    metadatas=[{"reward":1.0}]
)

# 긍정적 경험 (보상 - 1.0)
vectorstore.add_texts(
    ["State: 비밀번호 입력창 발견. Acction: 'password' 입력."],
    metadatas=[{"reward":-1.0}]
)

positive_retriever = vectorstore.as_retriever(
    search_kwargs = {"k":1, "filter":{"reward":1.0}}
)

negative_retriever = vectorstore.as_retriever(
    search_kwargs = {"k":1, "filter":{"reward":-1.0}}
)

# LLM에게 전달할 Memento 스타일의 프롬프트
template = """
당신은 목표를 달성해야 하는 AI 에이전트입니다.
과거의 성공 및 실패 사례를 참고하여 최적의 행동을 결정하세요.

### 현재 상태 ###
{state}

### 유사 상황 성공 사례 (모방할 것) ###
{positive_examples}

### 유사 상황 실패 사례 (회피할 것) ###
{negative_examples}

### 지시 ###
위 정보를 바탕으로 지금 당장 실행해야 할 행동(Action)을 한 문장으로 생성하세요.
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "positive_examples" : RunnableLambda(lambda x: positive_retriever.invoke(x["state"])),
        "negative_examples" : RunnableLambda(lambda x: negative_retriever.invoke(x["state"])),
        "state": RunnablePassthrough()
    } 
    | prompt
    | ChatOpenAI()
)

response = chain.invoke({"state":"State: 관리자 페이지의 비밀번호 입력창을 마주했다."})
print(response.content) 

