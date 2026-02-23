"""
RRF (Reciprocal Rank Fusion) 算法实现
用于融合多个检索结果列表
"""
import logging
from typing import List, Dict, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


def rrf_fusion(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    RRF (Reciprocal Rank Fusion) 融合算法
    
    公式: score(d) = Σ(1 / (k + rank(q, d)))
    其中:
    - d: 文档/chunk
    - q: 查询
    - rank(q, d): 文档在查询q的结果中的排名（从1开始）
    - k: 常数，通常取60
    
    Args:
        result_lists: 多个检索结果列表，每个列表包含 {'id': int, 'score': float, ...} 格式的字典
        k: RRF 常数，默认60
        top_k: 返回前K个结果
    
    Returns:
        融合后的结果列表，按RRF分数降序排列
    """
    if not result_lists:
        return []
    
    # 统计每个文档在所有查询结果中的排名和分数
    doc_scores = defaultdict(float)  # {doc_id: rrf_score}
    doc_data = {}  # {doc_id: doc_data} 保存第一个出现的完整文档数据
    
    # 遍历每个查询的结果列表
    for query_idx, result_list in enumerate(result_lists):
        if not result_list:
            continue
        
        # 遍历该查询结果中的每个文档
        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc.get('id')
            if not doc_id:
                # 如果没有id，尝试使用其他唯一标识
                doc_id = id(doc)  # 使用对象id作为fallback
                doc['id'] = doc_id
            
            # 计算RRF分数贡献
            rrf_contribution = 1.0 / (k + rank)
            doc_scores[doc_id] += rrf_contribution
            
            # 保存文档数据（如果第一次出现）
            if doc_id not in doc_data:
                doc_data[doc_id] = doc.copy()
                doc_data[doc_id]['rrf_score'] = 0.0  # 初始化
                doc_data[doc_id]['ranks'] = []  # 记录在各个查询中的排名
                doc_data[doc_id]['scores'] = []  # 记录原始分数
    
    # 更新每个文档的RRF分数和元数据
    for doc_id, rrf_score in doc_scores.items():
        if doc_id in doc_data:
            doc_data[doc_id]['rrf_score'] = rrf_score
    
    # 按RRF分数降序排序
    sorted_docs = sorted(
        doc_data.values(),
        key=lambda x: x.get('rrf_score', 0.0),
        reverse=True
    )
    
    # 返回前top_k个结果
    result = sorted_docs[:top_k]
    
    logger.info(
        f"[RRF Fusion] 融合 {len(result_lists)} 个查询结果，"
        f"共 {len(doc_scores)} 个唯一文档，返回 Top {len(result)} 个"
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
