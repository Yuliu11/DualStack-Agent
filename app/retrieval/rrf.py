"""
RRF (Reciprocal Rank Fusion) 算法实现
用于融合多个检索结果列表
"""
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def rrf_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 10,
    weights: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """
    RRF (Reciprocal Rank Fusion) 融合算法，支持各列表权重

    公式: score(d) = Σ weight_i * (1 / (k + rank(q_i, d)))
    其中 weight_i 默认为 1.0；用于区分全文检索(关键词)与向量检索(语义)等。

    Args:
        result_lists: 多个检索结果列表，每个列表包含 {'id': int, 'score': float, ...}
        k: RRF 常数，默认60
        top_k: 返回前K个结果
        weights: 各列表权重，与 result_lists 一一对应；如 [0.6, 0.4] 表示第一路 0.6、第二路 0.4

    Returns:
        融合后的结果列表，按RRF分数降序排列
    """
    if not result_lists:
        return []

    # 统计每个文档在所有查询结果中的加权分数
    doc_scores = defaultdict(float)
    doc_data = {}

    for query_idx, result_list in enumerate(result_lists):
        if not result_list:
            continue
        weight = 1.0
        if weights is not None and query_idx < len(weights):
            weight = weights[query_idx]

        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc.get('id')
            if not doc_id:
                doc_id = id(doc)
                doc['id'] = doc_id

            rrf_contribution = weight * (1.0 / (k + rank))
            doc_scores[doc_id] += rrf_contribution

            if doc_id not in doc_data:
                doc_data[doc_id] = doc.copy()
                doc_data[doc_id]['rrf_score'] = 0.0
                doc_data[doc_id]['ranks'] = []
                doc_data[doc_id]['scores'] = []

    for doc_id, rrf_score in doc_scores.items():
        if doc_id in doc_data:
            doc_data[doc_id]['rrf_score'] = rrf_score

    sorted_docs = sorted(
        doc_data.values(),
        key=lambda x: x.get('rrf_score', 0.0),
        reverse=True
    )
    result = sorted_docs[:top_k]

    logger.info(
        "[RRF Fusion] 融合 %d 个查询结果，共 %d 个唯一文档，返回 Top %d 个",
        len(result_lists), len(doc_scores), len(result)
    )
    return result


def merge_with_rrf(
    retrieval_results: List[Dict[str, Any]],
    k: int = 60,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    合并多个检索结果并使用RRF融合
    
    Args:
        retrieval_results: 检索结果列表，每个元素包含 {'query': str, 'chunks': List[Dict]}
        k: RRF常数
        top_k: 返回前K个结果
    
    Returns:
        融合后的chunk列表
    """
    # 提取所有chunk列表
    chunk_lists = []
    for result in retrieval_results:
        chunks = result.get('chunks', [])
        if chunks:
            chunk_lists.append(chunks)
    
    if not chunk_lists:
        return []
    
    # 执行RRF融合
    fused_chunks = rrf_fusion(chunk_lists, k=k, top_k=top_k)
    
    return fused_chunks
