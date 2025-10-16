# 🧠 AXDeepScholar

자기성찰형 AI 논문 연구용 딥리서치 에이전트 (Arxiv + Custom RAG + PostgreSQL Memory)

📘 프로젝트 개요

“AI 연구 논문 트렌드를 스스로 탐색·기억·분석하는 LangGraph 기반 딥리서치 에이전트”

1. 이 프로젝트는 단순한 논문 요약기가 아니라,
시간이 지날수록 더 똑똑해지는 연구자형 AI를 목표로 합니다.

2. LangChain의 Arxiv Tool을 통해 논문을 수집하고, 
직접 구현한 RAG 엔진으로 논문 내용을 임베딩·검색·요약합니다.

3. 결과는 PostgreSQL Memory에 저장되어,
 세션이 반복될수록 점점 더 정교한 트렌드 분석이 가능합니다.


🎯 목표
- Arxiv 논문을 기반으로 한 지식 축적형 RAG 시스템
- LangGraph 기반 3-Agent 구조 (Supervisor / Researcher / Reporter)
- Arxiv MCP(Function Calling)으로 실시간 논문 검색
- 직접 구현한 Custom RAG Retriever + Scorer
- PostgreSQL 기반 Persistent Memory (세션간 학습)

“시간이 흐를수록 성장하는 에이전트” 컨셉 실현


                             ┌────────────────────┐
                             │ SUPERVISOR AGENT  │
                             │────────────────────│
                             │  Load Memory (RDB) │
                             │  Plan Research     │
                             │  Trigger Agents    │
                             └──────────┬─────────┘
                                        │
                                        ▼
                     ┌────────────────────────────────┐
                     │        RESEARCHER AGENT         │
                     │────────────────────────────────│
                     │  Arxiv MCP 호출 (논문 검색)     │
                     │  Custom RAG Embedding 검색      │
                     │  Context Reorder + Score        │
                     │  Reflect (품질 평가/재요약)      │
                     └──────────┬──────────────────────┘
                                │
                                ▼
                    ┌──────────────────────────────┐
                    │        REPORTER AGENT        │
                    │──────────────────────────────│
                    │  Summarize + LLM Eval        │
                    │  Save Trend → PostgreSQL     │
                    │  Generate Trend Report (PDF) │
                    └──────────────────────────────┘

                    


| 구분                        | 사용 기술                                  |
| ------------------------- | -------------------------------------- |
| **Core Framework**        | LangGraph (StateGraph, Node Workflow)  |
| **LLM**                   | GPT-5 / GPT-5-mini                    |
| **Data Source**           | Arxiv API (논문 PDF + Metadata)          |
| **RAG Engine**            | Chroma / FAISS + BGE-M3 Embedding      |
| **MCP Tool**              | `arxiv_mcp` (논문 실시간 검색용 Function Tool) |
| **LangChain Integration** | Document Loaders, Text Splitters       |
| **Language**              | Python 3.10+                           |
| **Visualization**         | Matplotlib / Plotly (Trend Report)     |



deep_research_agent/
│
├─ agents/
│   ├─ supervisor.py      # 연구 흐름 관리 및 평가
│   ├─ researcher.py      # 논문 검색 + 요약 + Reflect
│   └─ reporter.py        # 리포트 생성 및 시각화
│
├─ tools/
│   ├─ arxiv_mcp.py       # Arxiv API MCP (function-calling)
│
├─ data/
│   ├─ raw/               # 원문 PDF 및 메타데이터
│   ├─ processed/         # Chunked text
│   └─ embeddings/        # VectorDB 저장소
│
├─ pipelines/
│   └─ deep_research_graph.py  # LangGraph StateGraph 정의
│
└─ outputs/
    └─ reports/            # Trend Reports (Markdown/PDF)


🔍 동작 과정

Supervisor Agent
- 연구 주제와 기간을 설정
- Macro-RAG(Global Memory)에서 과거 트렌드 조회

Researcher Agent
- Arxiv 최신 논문 검색
- PDF-RAG(Local VectorDB)로 관련 논문 의미 검색
- 자기성찰(Reflect)으로 요약 품질 점검

Reporter Agent
- 전체 결과를 통합

Faithfulness / Relevance 평가
- Markdown 및 PDF 리포트 생성