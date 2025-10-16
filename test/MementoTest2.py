from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_openai import OpenAIEmbeddings
from dummy import qa_pairs
from langchain_core.prompt_values  import ChatPromptValue
 
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

# pipeline 직접 구성
def run_model(prompt):
    """pipeline 없이 Falcon 직접 호출"""
    messages = [{"role": "user", "content": prompt}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=128, temperature=0.1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

llm = RunnableLambda(
    lambda x: {
        "content": run_model(
            x.messages[-1].content  
            if isinstance(x, ChatPromptValue) else str(x)  
        )
    }
)

embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
dummy_docs = [Document(page_content="Initialize FAISS memory.")]
vectorstore = FAISS.from_documents(dummy_docs, embeddings)

def evaluate_answer(answer: str, ground_truth: str) -> float:
    """정답이 포함되면 +1, 아니면 -1"""
    return 1.0 if ground_truth in answer else -1.0

positive_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3, "filter": {"reward": 1.0}}
)

negative_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3, "filter": {"reward": -1.0}}
)

prompt_template = """
You are an AI agent designed to infer the meaning of a given word.
Refer to your past experiences — successful cases (reward +1) and failed cases (reward -1) — 
to accurately predict the correct meaning.

### Successful Examples (Reference)
{positive_examples}

### Failed Examples (Avoid)
{negative_examples}

### Current Question
{state}

### Instruction
Provide the meaning of the word in one sentence using the format:
Action: [your answer]
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

chain = (
    {
        "positive_examples": RunnableLambda(
            lambda x: positive_retriever.invoke(x["state"])
        ),
        "negative_examples": RunnableLambda(
            lambda x: negative_retriever.invoke(x["state"])
        ),
        "state": RunnablePassthrough()
    }
    | prompt
    | llm
)

# === main loop ===
def run_ambiguous_test():
    print("\n--- 🧠 Memento Ambiguous QA Simulation (100개) ---\n")
    total_reward = 0

    for i, (question, answer) in enumerate(qa_pairs.items(), start=1):
        print(f"🔹 Q{i}: {question}")
        result = chain.invoke({"state": question})
        action = result["content"].split(":")[-1].strip()
        reward = evaluate_answer(action, answer)
        total_reward += reward

        print(f"🤖 AI 답변: {action}")
        print(f"🎯 정답: {answer} | 보상: {reward}")

        # Memory 기록 (Memento Write)
        vectorstore.add_documents([
            Document(page_content=f"State: '{question}' | Action: '{action}'",
                     metadata={"reward": reward})
        ])
        print("--------------------------\n")

    print(f"✅ 전체 시도: {len(qa_pairs)}개 | 총 보상: {total_reward}")

if __name__ == "__main__":
    run_ambiguous_test()
    # 메멘토 사용시 결과 - 1 
    # ✅ 전체 시도: 100개 | 총 보상: 12.0

    # 메멘토 없을때 - 1
    # ✅ 전체 시도: 104개 | 총 보상: -46.0


