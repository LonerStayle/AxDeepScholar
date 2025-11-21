import numpy as np
import re
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

def split_into_sentences(text: str) -> List[str]:
    """간단한 문장 분할기 (개행 + 구두점 기반)"""
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip()) > 10]
    return sents


def minmax_chunking_process(sentences, embeddings, fixed_threshold=0.6, c=0.9, init_constant=1.5):
    """
    문장 간의 의미적 유사도에 따라 문장들을 단락으로 묶습니다.
    
    매개변수 (Args):
    - sentences (list of str): 처리할 문장들의 리스트.
    - embeddings (np.array): 문장 임베딩 배열. 크기는 (문장 개수, 임베딩 차원).
    - fixed_threshold (float): 문장 병합 시 사용할 고정 유사도 임계값.
    - c (float): 유사도 임계값을 조정하기 위한 계수.
    - init_constant (float): 클러스터 크기가 1일 때 사용할 초기 비교 상수.
    
    반환값 (Returns):
    - list of list of str: 각 단락이 문장들의 리스트로 구성된 단락 리스트.
    """
    
    def sigmoid(x):
        """Sigmoid function for adjusting threshold based on cluster size."""
        return 1 / (1 + np.exp(-x))

    paragraphs = []
    current_paragraph = [sentences[0]]
    cluster_start, cluster_end = 0, 1
    pairwise_min = -float('inf')

    for i in range(1, len(sentences)):
        cluster_embeddings = embeddings[cluster_start:cluster_end]

        if cluster_end - cluster_start > 1:
            new_sentence_similarities = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
            adjusted_threshold = pairwise_min * c * sigmoid((cluster_end - cluster_start) - 1)
            new_sentence_similarity = np.max(new_sentence_similarities)
            pairwise_min = min(np.min(new_sentence_similarities), pairwise_min)
        else:
            adjusted_threshold = 0            
            pairwise_min = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
            new_sentence_similarity = init_constant * pairwise_min

        if new_sentence_similarity > max(adjusted_threshold, fixed_threshold):
            current_paragraph.append(sentences[i])
            cluster_end += 1
        else:
            paragraphs.append(current_paragraph)
            current_paragraph = [sentences[i]]
            cluster_start, cluster_end = i, i + 1
            pairwise_min = -float('inf')

    paragraphs.append(current_paragraph)
    return paragraphs

