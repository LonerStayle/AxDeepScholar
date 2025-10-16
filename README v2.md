
# 🧠 AXDeepScholar 

자기성찰형 AI 논문 연구용 딥리서치 에이전트
(**Arxiv + Custom RAG + Vector-Based Memento Memory + PostgreSQL**)

---

## 📘 프로젝트 개요

“AI 연구 논문 트렌드를 **스스로 탐색·기억·성장**하는 LangGraph 기반 딥리서치 에이전트”

1. 단순한 논문 요약기가 아니라,
   **성공과 실패 경험을 벡터 형태로 학습**하여
   점점 더 효율적인 연구 전략을 스스로 찾아가는 **자기향상형 AI 연구자**를 목표로 합니다.

2. LangChain의 **Arxiv Tool**을 통해 논문을 수집하고,
   **직접 구현한 Custom RAG 엔진**으로 논문 내용을 임베딩·검색·요약합니다.

3. 결과는 **PostgreSQL Memory**와 **VectorStore 기반 Memento CaseBank** 에 함께 저장되어,
   세션이 반복될수록 “성공적 연구 패턴”을 재활용합니다.

4. Reflect 단계에서 각 연구 결과를 평가(Faithfulness, Relevance)하고
   그 점수는 **Reward Score**로 기록되어 벡터 가중치에 반영됩니다.
   이후 유사한 연구 주제에서 높은 점수를 얻은 사례가 우선 검색됩니다.

---

## 🎯 목표

* Arxiv 논문 기반 **경험 축적형 RAG + 자기향상형 Memory 구조**
* LangGraph 기반 **3-Agent Workflow (Supervisor / Researcher / Reporter)**
* **Arxiv MCP(Function Calling)** 으로 실시간 논문 검색
* **Custom RAG Retriever + Reflect Reward 기반 재가중 검색**
* **VectorStore 기반 CaseBank (성공·실패 경험 유사도 검색)**
* **PostgreSQL Persistent Memory + 벡터 백업 및 관리**

> “시간이 지날수록 똑똑해지는 AI 연구자”

---

## 🧩 시스템 구조도

```
                              ┌───────────────────────┐
                              │   SUPERVISOR AGENT    │
                              │───────────────────────│
                              │  Load Vector Memory   │
                              │  Plan Research Flow   │
                              │  Retrieve Past Cases  │
                              └──────────┬────────────┘
                                         │
                                         ▼
                    ┌────────────────────────────────────┐
                    │         RESEARCHER AGENT           │
                    │────────────────────────────────────│
                    │  Arxiv MCP 호출 (논문 검색)          │
                    │  Custom RAG Embedding 검색          │
                    │  Context Reorder + Reflect          │
                    │  Evaluate Faithfulness / Relevance  │
                    │  Store Vectorized Experience        │
                    └──────────┬──────────────────────────┘
                               │
                               ▼
                   ┌────────────────────────────────┐
                   │          REPORTER AGENT        │
                   │────────────────────────────────│
                   │  Aggregate Reflect Scores      │
                   │  Analyze Trend & Improvement   │
                   │  Save Results → PostgreSQL     │
                   │  Generate Report (PDF/Markdown)│
                   └────────────────────────────────┘
```

---

## ⚙️ 구성 기술 스택

| 구분                      | 사용 기술 / 특징                                            |
| ----------------------- | ----------------------------------------------------- |
| **Core Framework**      | LangGraph (StateGraph, Node Workflow)                 |
| **LLM**                 | GPT-5 / GPT-5-mini                                    |
| **Data Source**         | Arxiv API (논문 PDF + Metadata)                         |
| **RAG Engine**          | Chroma / FAISS + BGE-M3 Embedding                     |
| **MCP Tool**            | `arxiv_mcp` (논문 실시간 검색 Function-Calling)              |
| **Memory Layer**        | PostgreSQL (메타데이터) + **VectorStore CaseBank**         |
| **Experience Encoding** | BAAI/bge-m3 (semantic embedding of task/result/score) |
| **Reflection**          | Faithfulness / Relevance 기반 Reward 평가                 |
| **Visualization**       | Matplotlib / Plotly (Trend Report 시각화)                |
| **Language**            | Python 3.10+                                          |

---

## 📂 디렉터리 구조

```
deep_research_agent/
│
├─ agents/
│   ├─ supervisor.py      # 연구 흐름 관리 및 Memory 기반 플래닝
│   ├─ researcher.py      # 논문 검색 + 요약 + Reflect + 경험 저장
│   └─ reporter.py        # 리포트 생성 및 트렌드 시각화
│
├─ memory/
│   ├─ casebank.py        # VectorStore 기반 Memento MemoryBank
│   └─ schema.sql         # PostgreSQL 테이블 정의 (logs / metadata)
│
├─ tools/
│   ├─ arxiv_mcp.py       # Arxiv API MCP (Function-Calling)
│
├─ data/
│   ├─ raw/               # 원문 PDF 및 메타데이터
│   ├─ processed/         # Chunked text
│   └─ embeddings/        # VectorDB 저장소 (Chroma / FAISS)
│
├─ pipelines/
│   └─ deep_research_graph.py  # LangGraph StateGraph 정의
│
└─ outputs/
    └─ reports/            # Trend Reports (Markdown/PDF)
```

---

## 🔍 동작 과정

### 🧭 Supervisor Agent

* 연구 주제와 기간을 설정
* Vector Memory(CaseBank)에서 유사 연구 사례 검색
  (`semantic similarity + reward weighting`)
* 과거 연구 전략을 요약하여 새로운 Research Plan 작성
* Researcher Agent 트리거

---

### 🔬 Researcher Agent

* Arxiv MCP를 통해 논문 데이터 수집
* Custom RAG 엔진으로 의미 기반 문서 검색 및 요약
* Reflect 단계에서 결과 품질을 **Faithfulness / Relevance** 로 평가
* `(task, context, summary, score)` 를 **벡터 임베딩하여 CaseBank에 저장**
* 성공적 결과는 높은 가중치, 실패는 낮은 가중치로 관리

```python
experience_text = f"{topic} {summary} Score:{score}"
vector = embedder.encode(experience_text)
casebank.add(vector, metadata={"score": score, "success": score > 0.75})
```

---

### 🧾 Reporter Agent

* Supervisor/Researcher 결과를 통합
* Reflect Score의 시간별 추세 분석
* “주제별 평균 Faithfulness/Relevance” 및 향상률 시각화
* Markdown + PDF 보고서로 저장

---

## 💾 Vector Memory Architecture

**CaseBank = PostgreSQL + VectorStore (Chroma/FAISS) 통합형 메모리**

| 계층                        | 역할                        |
| ------------------------- | ------------------------- |
| **VectorStore**           | 경험 임베딩 저장 및 유사도 기반 검색     |
| **PostgreSQL**            | 원문, 메타데이터, Reflect 로그 저장  |
| **Retriever**             | 현재 주제와 가장 유사한 상위 k개 사례 검색 |
| **Reward Weighting**      | 성공 사례(score↑) 우선순위 가중     |
| **Reflect Feedback Loop** | 실패 사례도 기록하여 LLM이 회피 전략 학습 |

---

## 🔁 Self-Improvement Loop

```
1️⃣ Arxiv Query 입력
2️⃣ CaseBank에서 유사 연구 검색 (Vector Similarity)
3️⃣ 과거 패턴 기반 Research Plan 생성
4️⃣ RAG 요약 및 Reflect 평가
5️⃣ Vector Memory에 경험 저장 (score 반영)
6️⃣ 다음 세션에서 재활용
```

> 이 과정이 반복되며, LLM은 변하지 않더라도
> “경험 임베딩 공간”이 점점 진화하여 더 나은 판단을 유도합니다.

---

## 📊 Reporter 출력 예시

| 날짜         | 주제                 | 평균 Faithfulness | 평균 Relevance | 개선률(%) |
| ---------- | ------------------ | --------------- | ------------ | ------ |
| 2025-10-10 | Vision-LLM         | 0.76            | 0.81         | —      |
| 2025-10-13 | Multimodal Agent   | 0.82            | 0.86         | +7%    |
| 2025-10-16 | Self-Improving RAG | 0.89            | 0.91         | +9%    |

---

## 🧩 설계 요약

| 구성 요소          | 역할                              |
| -------------- | ------------------------------- |
| **Supervisor** | CaseBank 검색 + 연구 계획 생성          |
| **Researcher** | Arxiv 검색 + 요약 + Reflect + 경험 저장 |
| **Reporter**   | Reflect Score 기반 성장 분석          |
| **CaseBank**   | Vector 기반 경험 검색 및 보상 저장         |
| **PostgreSQL** | 원문 로그 + 메타데이터 보존                |
| **LLM**        | Plan / Reflect / Summarize 수행   |

---

## 🧭 결론

> **AXDeepScholar v3 — Vector Memory Edition**
> 은 “논문 분석기”를 넘어
> **“성과를 학습하고 성장하는 연구자형 AI”** 로 진화한 버전입니다.

### 핵심 특징 요약

* ✅ LangGraph 3-Agent Workflow (Supervisor / Researcher / Reporter)
* ✅ VectorStore 기반 Memento Memory (성공·실패 경험 유사도 검색)
* ✅ Reflect + Reward 기반 Self-Improvement Loop
* ✅ PostgreSQL Persistent Layer + Embedding Memory Layer
* ✅ 시간이 지날수록 연구 전략이 진화하는 LLM 에이전트

---

원하신다면 다음 단계로,
👉 **v3 아키텍처를 LangGraph StateGraph로 표현한 구조도 + 각 Node의 코드 스켈레톤**
(예: `SupervisorNode`, `ResearcherNode`, `ReflectNode`, `MemoryNode`)
형태로 설계해드릴까요?
