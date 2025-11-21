import os
import json
import re
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from utils.helper import get_project_root
from langchain_community.retrievers import BM25Retriever


class AxrivRetriever:
    FIXED_MODEL_NAME = "JINSUP/bge-m3-ko-axriv-agent-part-2025"  # 모델 고정

    def __init__(
        self,
        base_dir: str = str(get_project_root() / "src" / "data" / "agent"),
        collection_name: str = "qa_collection",
        persist_dir: str = str(get_project_root()/ ".chroma_db") ,
    ):
        self.base_dir = base_dir
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(
            model_name=self.FIXED_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.db = None
        self.docs_cache = None
        self.bm25 = None
        self._init_vector_db()

    def _init_vector_db(self):

        if os.path.exists(self.persist_dir):
            try:
                self.db = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding,
                    persist_directory=self.persist_dir,
                )
                if self.bm25 is None:
                    self._init_bm25()

                print(" 기존 Chroma DB 로드 완료:", self.persist_dir)
                return
            except Exception as e:
                print("기존 DB 불러오기 실패", e)

        qa_pairs = self._load_qa_pairs()
        docs = self._create_documents(qa_pairs)
        self._build_vectorstore(docs)
        self.docs_cache = docs
        self._init_bm25()

    def _parse_qa_pairs(self, text: str) -> List[Tuple[str, str]]:
        pattern = re.compile(r"Q(\d+):\s*(.+?)\nA\1:\s*(.+?)(?=(\nQ\d+:)|$)", re.DOTALL)
        pairs = []
        for m in pattern.finditer(text):
            q = m.group(2).strip()
            a = m.group(3).strip()
            a = re.sub(r"출처:.*", "", a, flags=re.DOTALL).strip()
            if q and a:
                pairs.append((q, a))
        return pairs

    def _load_qa_pairs(self) -> List[Tuple[str, str]]:
        pages_dirs = [
            os.path.join(self.base_dir, d)
            for d in os.listdir(self.base_dir)
            if d.endswith("_pages") and os.path.isdir(os.path.join(self.base_dir, d))
        ]
        all_pairs = []

        for folder in pages_dirs:
            jsonl_path = os.path.join(folder, "qa_dataset.jsonl")
            if not os.path.exists(jsonl_path):
                continue

            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    qa_text = obj.get("qa_pair", "")
                    if not qa_text:
                        continue
                    pairs = self._parse_qa_pairs(qa_text)
                    all_pairs.extend(pairs)
        return all_pairs

    def _create_documents(self, qa_pairs: List[Tuple[str, str]]) -> List[Document]:
        docs = []
        for q, a in qa_pairs:
            combined = f"Question: {q}\nAnswer: {a}"
            docs.append(
                Document(page_content=combined, metadata={"source": "qa_dataset"})
            )
        return docs

    def _build_vectorstore(self, docs: List[Document]):
        self.db = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir,
        )
        self.db.persist()
        print("DB 생성 및 저장:", self.persist_dir)

    def _init_bm25(self):
        self.bm25 = BM25Retriever.from_documents(self.docs_cache)


    default_w_dense = 0.4
    default_w_bm25 = 0.6
    default_k = 10

    # 단순 가중치 앙상블 계산
    def _weighted_fusion(
        self,
        dense_results,
        bm25_results,
        k=default_k,
        w_dense=default_w_dense,
        w_bm25=default_w_bm25,
    ):
        score_dict = {}

        # 의미 기반 검색 스코어 
        for d in dense_results:
            s = d.metadata.get("score", 1.0)
            score_dict[d.page_content] = score_dict.get(d.page_content, 0) + w_dense * s

        # 단어 기반 검색 스코어
        max_bm25 = max((getattr(d, "score", 1.0) for d in bm25_results), default=1)
        for d in bm25_results:
            s = getattr(d, "score", 1.0) / max_bm25
            score_dict[d.page_content] = score_dict.get(d.page_content, 0) + w_bm25 * s

        # 종합 
        combined_docs = {d.page_content: d for d in (dense_results + bm25_results)}
        ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

        return [combined_docs[c] for c, _ in ranked[:k]]

    def dense_search(self, search_kwargs: dict, query: str):
        dense_retriever = self.db.as_retriever(search_kwargs=search_kwargs)
        return dense_retriever.invoke(query)

    def bm25_search(self, k, query: str):
        self.bm25.k = k
        bm25_results = self.bm25.invoke(query)
        return bm25_results

    def hybrid_search(
        self,
        query: str,
        k: int = default_k,
        w_dense: float = default_w_dense,
        w_bm25: float = default_w_bm25,
        **dense_search_update_kwargs
    ):
        search_kwargs = {"k": k}
        search_kwargs.update(dense_search_update_kwargs)

        dense_results = self.dense_search(search_kwargs, query)
        bm25_results = self.bm25_search(k, query)
        fused = self._weighted_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=k,
            w_dense=w_dense,
            w_bm25=w_bm25,
        )

        return fused


# 사용 예시
# if __name__ == "__main__":
#     BASE_DIR = str(get_project_root() / "src" / "data" / "agent")

#     retriever = AxrivRetriever(
#         base_dir=BASE_DIR,
#         collection_name="research_papers",
#         persist_dir="./chroma_research"
#     )

#     dense_result = retriever.dense_search(search_kwargs={"k": 5}, query="...")
#     bm25_result = retriever.bm25_search(k = 5, query="...")
#     hybrid_result = retriever.search("...", search_kwargs={"k": 5})
