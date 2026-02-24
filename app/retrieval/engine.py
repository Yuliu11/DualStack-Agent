"""
混合检索引擎
实现向量搜索（FAISS/Qdrant）和BM25关键词搜索，并进行结果融合
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.config import yaml_config
from app.storage.db_manager import db_manager
from app.storage.models import Chunk, Document, Embedding
from app.llm import llm_service
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_service import BM25Service
from app.retrieval.rrf import rrf_fusion

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """混合检索引擎（向量搜索 + BM25 + RRF融合）"""
    
    def __init__(self):
        """初始化检索引擎"""
        self.retrieval_config = yaml_config.get("retrieval", {})
        self.default_top_k = self.retrieval_config.get("default_top_k", 5)
        
        # 向量存储配置
        vector_config = yaml_config.get("vector_store", {})
        self.vector_store_type = vector_config.get("default", "faiss")
        self.index_path = vector_config.get("faiss", {}).get("index_path", "./data/faiss_index")
        
        # 初始化向量存储
        self.vector_store = VectorStore(
            vector_store_type=self.vector_store_type,
            index_path=self.index_path
        )
        
        # 初始化BM25服务
        self.bm25_service = BM25Service()
        self._bm25_initialized = False
        
        # 嵌入模型配置
        self.embed_model = yaml_config.get("models", {}).get("default_embedding", "text-embedding-v2")
    
    async def _ensure_bm25_index(self, tenant_id: int = 0):
        """确保BM25索引已构建"""
        if self._bm25_initialized:
            return
        
        try:
            # 从数据库加载所有chunks
            async with db_manager.get_session() as session:
                stmt = select(Chunk).where(
                    and_(
                        Chunk.tenant_id == tenant_id,
                        Chunk.is_deleted == False
                    )
                )
                result = await session.execute(stmt)
                chunks = result.scalars().all()
            
            # 转换为字典格式
            chunks_data = [
                {
                    'id': chunk.id,
                    'content': chunk.content,
                    'document_id': chunk.document_id
                }
                for chunk in chunks
            ]
            
            # 构建BM25索引
            await self.bm25_service.build_index(chunks_data, tenant_id)
            self._bm25_initialized = True
            logger.info(f"BM25 index initialized with {len(chunks_data)} chunks")
        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
    
    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        获取查询的嵌入向量
        
        Args:
            query: 查询文本
        
        Returns:
            嵌入向量，如果失败返回None（不会抛出异常，确保可以降级到BM25）
        """
        try:
            # 调用阿里云百炼的embedding API
            from app.config.config import settings
            from openai import AsyncOpenAI
            
            # 使用阿里云百炼的embedding模型
            embed_model = self.embed_model
            
            # 验证模型名称配置
            if not embed_model:
                logger.warning("Embedding model not configured, skipping vector search")
                return None
            
            # 创建DashScope客户端（使用OpenAI兼容接口）
            # 阿里云百炼的embedding API使用兼容模式路径
            # 官方路径：https://dashscope.aliyuncs.com/compatible-mode/v1
            dashscope_base_url = settings.dashscope_base_url
            
            # 确保使用兼容模式路径（用于 embedding API）
            if "/compatible-mode" not in dashscope_base_url:
                if dashscope_base_url.endswith("/v1"):
                    # 如果已经是 /v1，替换为 /compatible-mode/v1
                    dashscope_base_url = dashscope_base_url.replace("/v1", "/compatible-mode/v1")
                elif not dashscope_base_url.endswith("/compatible-mode/v1"):
                    # 如果既不是 /v1 也不是 /compatible-mode/v1，添加兼容模式路径
                    dashscope_base_url = f"{dashscope_base_url.rstrip('/')}/compatible-mode/v1"
            
            dashscope_client = AsyncOpenAI(
                api_key=settings.dashscope_api_key,
                base_url=dashscope_base_url,
                timeout=10.0  # 设置超时时间，避免长时间等待
            )
            
            # 调用embedding API
            # 模型名称：text-embedding-v2（从 config.yaml 读取，已验证无特殊字符）
            response = await dashscope_client.embeddings.create(
                model=embed_model,
                input=query
            )
            
            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                logger.debug(f"Got embedding for query, dimension: {len(embedding)}")
                return embedding
            else:
                logger.warning("Empty embedding response")
                return None
                
        except Exception as e:
            # 捕获所有异常，不抛出，确保可以降级到BM25搜索
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                logger.warning(
                    f"Embedding API 404 error (model '{self.embed_model}' not found): {e}. "
                    f"Please check config.yaml embedding_model configuration (should be 'text-embedding-v2'). "
                    f"Will fallback to BM25 search."
                )
            elif "401" in error_msg or "unauthorized" in error_msg.lower():
                logger.error(
                    f"Embedding API authentication failed: {e}. "
                    f"Please check DASHSCOPE_API_KEY. Will fallback to BM25 search."
                )
            elif "timeout" in error_msg.lower():
                logger.warning(
                    f"Embedding API timeout: {e}. Will fallback to BM25 search."
                )
            else:
                logger.error(
                    f"Failed to get query embedding: {e}. Will fallback to BM25 search."
                )
            return None  # 返回None，让系统自动降级到BM25搜索
    
    async def vector_search(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: int = 0
    ) -> List[Dict[str, Any]]:
        """
        向量搜索
        
        Args:
            query: 查询文本
            top_k: 返回Top-K个结果
            tenant_id: 租户ID
        
        Returns:
            搜索结果列表，每个元素包含 {'id': chunk_id, 'score': similarity_score, 'document_id': int}
        """
        try:
            # 获取查询向量
            query_vector = await self._get_query_embedding(query)
            if query_vector is None:
                logger.warning("Query embedding failed, skipping vector search")
                return []
            
            # 向量搜索
            vector_results = await self.vector_store.search(
                query_vector=query_vector,
                top_k=top_k,
                tenant_id=tenant_id
            )
            
            # 获取完整的chunk信息（包括document_id）
            if vector_results:
                chunk_ids = [r['id'] for r in vector_results]
                async with db_manager.get_session() as session:
                    stmt = select(Chunk).where(
                        and_(
                            Chunk.id.in_(chunk_ids),
                            Chunk.tenant_id == tenant_id,
                            Chunk.is_deleted == False
                        )
                    )
                    result = await session.execute(stmt)
                    chunks = result.scalars().all()
                    
                    chunk_map = {chunk.id: chunk for chunk in chunks}
                    
                    # 补充document_id等信息
                    for vec_result in vector_results:
                        chunk_id = vec_result['id']
                        chunk = chunk_map.get(chunk_id)
                        if chunk:
                            vec_result['document_id'] = chunk.document_id
                            vec_result['content'] = chunk.content[:200]  # 预览
                        else:
                            vec_result['document_id'] = None
                            vec_result['content'] = None
            
            logger.info(f"Vector search returned {len(vector_results)} results")
            return vector_results
        except Exception as e:
            # 向量搜索失败（如404错误），降级处理
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                logger.warning(f"Vector search failed (404/not found): {e}, will fallback to BM25")
            else:
                logger.error(f"Vector search failed: {e}, will fallback to BM25")
            return []  # 返回空列表，让系统降级到BM25搜索
    
    async def keyword_search(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: int = 0
    ) -> List[Dict[str, Any]]:
        """
        BM25关键词搜索
        
        Args:
            query: 查询文本
            top_k: 返回Top-K个结果
            tenant_id: 租户ID
        
        Returns:
            搜索结果列表，每个元素包含 {'id': chunk_id, 'score': bm25_score, 'document_id': int}
        """
        # 确保BM25索引已构建
        await self._ensure_bm25_index(tenant_id)
        
        # BM25搜索
        bm25_results = await self.bm25_service.search(
            query=query,
            top_k=top_k,
            tenant_id=tenant_id
        )
        
        # 获取完整的chunk信息（包括document_id）
        if bm25_results:
            chunk_ids = [r['id'] for r in bm25_results]
            async with db_manager.get_session() as session:
                stmt = select(Chunk).where(
                    and_(
                        Chunk.id.in_(chunk_ids),
                        Chunk.tenant_id == tenant_id,
                        Chunk.is_deleted == False
                    )
                )
                result = await session.execute(stmt)
                chunks = result.scalars().all()
                
                chunk_map = {chunk.id: chunk for chunk in chunks}
                
                # 补充document_id等信息
                for bm25_result in bm25_results:
                    chunk_id = bm25_result['id']
                    chunk = chunk_map.get(chunk_id)
                    if chunk:
                        bm25_result['document_id'] = chunk.document_id
                        bm25_result['content'] = chunk.content[:200]  # 预览
                    else:
                        bm25_result['document_id'] = None
                        bm25_result['content'] = None
        
        logger.info(f"BM25 search returned {len(bm25_results)} results")
        return bm25_results

    async def _filter_by_year(
        self,
        chunk_ids: List[int],
        query_years: List[int],
        tenant_id: int = 0,
    ) -> Optional[set]:
        """
        按年份过滤 chunk：并集过滤 + 空年份通过 + 内容补偿。

        - 并集：query_years=[2022, 2024] 时保留 metadata.year 为 2022 或 2024 的片段。
        - 空年份通过：metadata 中无年份的片段一律保留。
        - 内容补偿：metadata 不匹配时，若片段正文包含目标年份字符串（如 "2022"）则保留。

        Returns:
            允许的 chunk id 集合（空输入时返回空集合）。
        """
        if not chunk_ids or not query_years:
            return set()
        years_set = set(query_years)
        year_strs = [str(y) for y in query_years]

        async with db_manager.get_session() as session:
            stmt = (
                select(Chunk.id, Document.year, Chunk.content)
                .join(Document, Chunk.document_id == Document.id)
                .where(
                    Chunk.id.in_(chunk_ids),
                    Chunk.tenant_id == tenant_id,
                    Chunk.is_deleted == False,
                )
            )
            result = await session.execute(stmt)
            rows = result.fetchall()

        allowed = set()
        for row in rows:
            cid, doc_year, content = row[0], row[1], (row[2] or "")
            # 并集：metadata 年份在 query_years 中则保留
            if doc_year is not None and doc_year in years_set:
                allowed.add(cid)
                continue
            # 空年份通过：缺失年份信息则保留
            if doc_year is None:
                allowed.add(cid)
                continue
            # 内容补偿：正文包含目标年份字符串则保留
            if any(ys in content for ys in year_strs):
                allowed.add(cid)
        return allowed

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        tenant_id: int = 0,
        years: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        混合检索接口（向量搜索 + BM25 + 加权RRF融合），支持按年份过滤。

        Args:
            query: 查询文本
            top_k: 返回的 Top-K 数量
            tenant_id: 租户ID
            years: 可选，仅保留文档年份在此列表中的 chunk（如 [2023, 2024]）

        Returns:
            检索结果字典，包含 chunks 列表
        """
        try:
            # 1. 向量搜索
            vector_results = []
            try:
                vector_results = await self.vector_search(
                    query=query,
                    top_k=top_k,
                    tenant_id=tenant_id
                )
            except Exception as e:
                logger.warning(f"Vector search failed, will use BM25 only: {e}")
                vector_results = []

            # 2. BM25 关键词搜索
            bm25_results = []
            try:
                bm25_results = await self.keyword_search(
                    query=query,
                    top_k=top_k,
                    tenant_id=tenant_id
                )
            except Exception as e:
                logger.error(f"BM25 search also failed: {e}")
                bm25_results = []

            if not vector_results and not bm25_results:
                logger.warning(f"Both vector and BM25 search failed for query: {query[:50]}...")

            # 2.5 年份过滤（并集 + 空年份通过 + 内容补偿 + 过少时降级）
            if years:
                all_ids = list(dict.fromkeys(
                    [r["id"] for r in vector_results] + [r["id"] for r in bm25_results]
                ))
                if all_ids:
                    allowed_ids = await self._filter_by_year(
                        chunk_ids=all_ids,
                        query_years=years,
                        tenant_id=tenant_id,
                    )
                    if len(allowed_ids) < 3:
                        logger.warning(
                            "[Year Filter] 过滤后片段过少(%d)，降级保留原始检索 Top %d",
                            len(allowed_ids), min(5, len(all_ids)),
                        )
                        allowed_ids = set(all_ids)
                    vector_results = [r for r in vector_results if r["id"] in allowed_ids]
                    bm25_results = [r for r in bm25_results if r["id"] in allowed_ids]
                    logger.info(
                        "[Year Filter] query_years=%s(并集), allowed chunks: %d",
                        years, len(allowed_ids),
                    )

            # 3. 加权 RRF 融合（向量 0.6，BM25 0.4，保证财务数据精确度）
            rrf_k = self.retrieval_config.get("rrf_k", 60)
            vector_weight = self.retrieval_config.get("rrf_vector_weight", 0.6)
            bm25_weight = self.retrieval_config.get("rrf_bm25_weight", 0.4)

            if vector_results and bm25_results:
                fused_results = rrf_fusion(
                    result_lists=[vector_results, bm25_results],
                    k=rrf_k,
                    top_k=top_k,
                    weights=[vector_weight, bm25_weight],
                )
                logger.info(
                    "[Hybrid Search] Vector: %d, BM25: %d, Fused: %d (weights %.1f:%.1f)",
                    len(vector_results), len(bm25_results), len(fused_results),
                    vector_weight, bm25_weight,
                )
            elif vector_results:
                fused_results = vector_results[:top_k]
                logger.info("[Hybrid Search] Vector only: %d results", len(fused_results))
            elif bm25_results:
                fused_results = bm25_results[:top_k]
                logger.info("[Hybrid Search] BM25 only: %d results", len(fused_results))
            else:
                # 没有搜索结果
                fused_results = []
                logger.warning(f"[Hybrid Search] No results for query: {query[:50]}...")
            
            # 4. 获取完整的chunk数据
            if fused_results:
                chunk_ids = [r['id'] for r in fused_results]
                async with db_manager.get_session() as session:
                    stmt = select(Chunk).where(
                        and_(
                            Chunk.id.in_(chunk_ids),
                            Chunk.tenant_id == tenant_id,
                            Chunk.is_deleted == False
                        )
                    )
                    result = await session.execute(stmt)
                    chunks = result.scalars().all()
                    
                    chunk_map = {chunk.id: chunk for chunk in chunks}
                    
                    # 构建最终结果
                    final_chunks = []
                    for fused_result in fused_results:
                        chunk_id = fused_result['id']
                        chunk = chunk_map.get(chunk_id)
                        if chunk:
                            final_chunks.append({
                                'id': chunk.id,
                                'content': chunk.content,
                                'score': fused_result.get('score', 0.0),
                                'document_id': chunk.document_id,
                                'page_start': chunk.page_start,
                                'page_end': chunk.page_end,
                                'section_path': chunk.section_path,
                                'is_table': chunk.is_table
                            })
                    
                    return {
                        'chunks': final_chunks,
                        'query': query,
                        'top_k': top_k,
                        'vector_count': len(vector_results),
                        'bm25_count': len(bm25_results),
                        'fused_count': len(final_chunks)
                    }
            else:
                return {
                    'chunks': [],
                    'query': query,
                    'top_k': top_k,
                    'vector_count': 0,
                    'bm25_count': 0,
                    'fused_count': 0
                }
                
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return {
                'chunks': [],
                'query': query,
                'top_k': top_k,
                'error': str(e)
            }
