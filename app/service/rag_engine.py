"""
RAG 核心服务
协调检索引擎和 LLM 服务，实现完整的 RAG 流程
"""
import hashlib
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.db_manager import db_manager
from app.storage.models import QARequest, QACitation, Chunk
from app.llm import llm_service
from app.config.config import yaml_config
from app.pipeline.query import query_rewriter
from app.retrieval.rrf import merge_with_rrf

# 配置日志
import logging
logger = logging.getLogger(__name__)


class RAGService:
    """RAG 服务类，协调检索和生成"""
    
    def __init__(self):
        """初始化 RAG 服务"""
        self.default_tenant_id = 0
        self.retrieval_config = yaml_config.get("retrieval", {})
        self.default_top_k = self.retrieval_config.get("default_top_k", 5)
        
        # 延迟导入检索引擎（避免循环依赖）
        self._retrieval_engine = None
    
    @property
    def retrieval_engine(self):
        """延迟加载检索引擎"""
        if self._retrieval_engine is None:
            from app.retrieval.engine import RetrievalEngine
            self._retrieval_engine = RetrievalEngine()
        return self._retrieval_engine
    
    def calculate_query_hash(self, query: str) -> str:
        """计算查询的 SHA256 哈希值"""
        return hashlib.sha256(query.encode('utf-8')).hexdigest()
    
    async def create_qa_request(
        self,
        session: AsyncSession,
        query: str,
        tenant_id: int = 0,
        user_id: Optional[int] = None
    ) -> QARequest:
        """创建问答请求记录"""
        query_hash = self.calculate_query_hash(query)
        
        qa_request = QARequest(
            tenant_id=tenant_id,
            user_id=user_id,
            query_text=query,
            query_hash=query_hash,
            answer_policy="cite_required"
        )
        session.add(qa_request)
        await session.flush()  # 获取 ID
        return qa_request
    
    async def get_chunks_by_ids(
        self,
        session: AsyncSession,
        chunk_ids: List[int],
        tenant_id: int = 0
    ) -> List[Dict[str, Any]]:
        """根据 Chunk ID 列表获取 Chunk 数据"""
        if not chunk_ids:
            return []
        
        stmt = select(Chunk).where(
            and_(
                Chunk.id.in_(chunk_ids),
                Chunk.tenant_id == tenant_id,
                Chunk.is_deleted == False
            )
        )
        
        result = await session.execute(stmt)
        chunks = result.scalars().all()
        
        return [
            {
                'id': chunk.id,
                'content': chunk.content,
                'page_start': chunk.page_start,
                'page_end': chunk.page_end,
                'section_path': chunk.section_path,
                'is_table': chunk.is_table,
                'table_id': chunk.table_id
            }
            for chunk in chunks
        ]
    
    async def save_citations(
        self,
        session: AsyncSession,
        request_id: int,
        chunk_ids: List[int],
        chunks_data: List[Dict[str, Any]]
    ):
        """保存引用关系"""
        if not chunk_ids:
            return
        
        # 创建 chunk_id 到 chunk 数据的映射
        chunk_map = {chunk['id']: chunk for chunk in chunks_data}
        
        # 为每个引用的 chunk 创建 citation 记录
        for rank_no, chunk_id in enumerate(chunk_ids, start=1):
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            
            # 提取引用文本（取前 200 字符）
            quote_text = chunk.get('content', '')[:200] if chunk.get('content') else None
            
            citation = QACitation(
                request_id=request_id,
                chunk_id=chunk_id,
                rank_no=rank_no,
                quote_text=quote_text
            )
            session.add(citation)
    
    async def query(
        self,
        query: str,
        tenant_id: int = 0,
        user_id: Optional[int] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        RAG 查询主流程
        
        Args:
            query: 用户查询
            tenant_id: 租户ID
            user_id: 用户ID
            top_k: 检索返回的 Top-K 数量
        
        Returns:
            查询结果字典
        """
        top_k = top_k or self.default_top_k
        start_time = time.time()
        request_id = None
        
        try:
            # 1. 创建问答请求记录
            async with db_manager.get_session() as session:
                qa_request = await self.create_qa_request(
                    session,
                    query,
                    tenant_id,
                    user_id
                )
                request_id = qa_request.id
                await session.commit()
            
            # 2. Query 扩写/改写
            rewrite_result = await query_rewriter.rewrite_query(
                query=query,
                conversation_history=None,  # TODO: 可以从上下文获取对话历史
                tenant_id=tenant_id
            )
            
            rewritten_queries = rewrite_result.get('rewritten_queries', [query])
            primary_query = rewrite_result.get('primary_query', query)
            
            # 保存改写结果到数据库
            async with db_manager.get_session() as session:
                stmt = select(QARequest).where(QARequest.id == request_id)
                result = await session.execute(stmt)
                qa_request = result.scalar_one()
                qa_request.expanded_query_json = {
                    'original_query': query,
                    'rewritten_queries': rewritten_queries,
                    'primary_query': primary_query,
                    'rewrite_type': rewrite_result.get('rewrite_type', 'original'),
                    'rewrite_details': rewrite_result.get('rewrite_details', {})
                }
                await session.commit()
            
            # 3. 多路召回：使用改写后的查询进行检索
            # 对每个改写后的查询进行检索，各取Top 10个结果
            retrieval_config = yaml_config.get("retrieval", {})
            multi_route_top_k = retrieval_config.get("multi_route_top_k", 10)  # 每个查询取Top 10
            rrf_k = retrieval_config.get("rrf_k", 60)  # RRF常数k
            
            retrieval_results = []
            
            for rewritten_query in rewritten_queries:
                logger.info(f"[Multi-Route Retrieval] 查询: {rewritten_query[:50]}...")
                
                retrieval_result = await self.retrieval_engine.retrieve(
                    query=rewritten_query,
                    top_k=multi_route_top_k,  # 每个查询取Top 10
                    tenant_id=tenant_id
                )
                
                retrieval_results.append({
                    'query': rewritten_query,
                    'chunks': retrieval_result.get('chunks', [])
                })
                
                logger.info(
                    f"[Multi-Route Retrieval] 查询 '{rewritten_query[:30]}...' "
                    f"检索到 {len(retrieval_result.get('chunks', []))} 个结果"
                )
            
            # 4. RRF融合：将多个查询结果进行RRF融合
            logger.info(
                f"[RRF Fusion] 开始融合 {len(retrieval_results)} 个查询结果，"
                f"RRF常数k={rrf_k}"
            )
            
            fused_chunks = merge_with_rrf(
                retrieval_results=retrieval_results,
                k=rrf_k,
                top_k=top_k * 2  # 融合后取更多候选，供重排序使用
            )
            
            logger.info(
                f"[RRF Fusion] 融合完成，得到 {len(fused_chunks)} 个候选chunks"
            )
            
            # 5. 重排序（可选）：如果配置了reranker，进行精排
            rerank_enabled = retrieval_config.get("rerank_enabled", False)
            rerank_top_k = retrieval_config.get("rerank_top_k", 5)
            
            if rerank_enabled:
                logger.info(f"[Rerank] 开始重排序，从 {len(fused_chunks)} 个候选中选择 Top {rerank_top_k}")
                # TODO: 实现reranker调用（如BGE-Reranker）
                # 这里先使用简单的RRF分数排序
                reranked_chunks = fused_chunks[:rerank_top_k]
                logger.info(f"[Rerank] 重排序完成，选择 Top {len(reranked_chunks)} 个chunks")
            else:
                # 如果没有reranker，直接使用RRF融合后的前top_k个
                reranked_chunks = fused_chunks[:top_k]
                logger.info(f"[No Rerank] 直接使用RRF融合结果，选择 Top {len(reranked_chunks)} 个chunks")
            
            # 输出验证：记录融合后的结果分布
            if reranked_chunks:
                rrf_scores = [chunk.get('rrf_score', 0.0) for chunk in reranked_chunks]
                logger.info(
                    f"[Output Validation] 最终选择 {len(reranked_chunks)} 个chunks，"
                    f"RRF分数范围: [{min(rrf_scores):.4f}, {max(rrf_scores):.4f}], "
                    f"平均: {sum(rrf_scores)/len(rrf_scores):.4f}"
                )
            
            retrieved_chunks = reranked_chunks
            
            if not retrieved_chunks:
                # 没有检索到相关文档
                async with db_manager.get_session() as session:
                    stmt = select(QARequest).where(QARequest.id == request_id)
                    result = await session.execute(stmt)
                    qa_request = result.scalar_one()
                    
                    qa_request.answer_text = "抱歉，未找到相关信息。"
                    qa_request.is_refused = True
                    qa_request.confidence = 0.0
                    qa_request.latency_ms = int((time.time() - start_time) * 1000)
                    qa_request.retrieval_json = {"chunks_count": 0}
                    
                    await session.commit()
                
                return {
                    'success': True,
                    'request_id': request_id,
                    'answer': "抱歉，未找到相关信息。",
                    'confidence': 0.0,
                    'is_refused': True,
                    'citations': [],
                    'latency_ms': int((time.time() - start_time) * 1000)
                }
            
            # 4. 获取完整的 Chunk 数据
            chunk_ids = [chunk['id'] for chunk in retrieved_chunks]
            
            async with db_manager.get_session() as session:
                chunks_data = await self.get_chunks_by_ids(
                    session,
                    chunk_ids,
                    tenant_id
                )
            
            # 5. 上下文构建：将检索到的片段格式化为带 [Chunk ID] 的文本块
            # （LLMService.generate_answer 内部会处理格式）
            
            # 6. LLM 生成：调用 LLMService.generate_answer
            llm_result = await llm_service.generate_answer(
                query=query,
                context_chunks=chunks_data,
                request_id=request_id
            )
            
            answer = llm_result.get('answer', '')
            confidence = llm_result.get('confidence', 0.0)
            is_refused = llm_result.get('is_refused', False)
            model_name = llm_result.get('model_name', 'unknown')
            extracted_chunk_ids = llm_result.get('chunk_ids', [])
            
            # 7. 存入引用表：将模型实际引用的 chunk_ids 存入 qa_citations 表
            async with db_manager.get_session() as session:
                await self.save_citations(
                    session,
                    request_id,
                    extracted_chunk_ids,
                    chunks_data
                )
                
                # 8. 更新请求状态：将最终生成的答案写回 qa_requests
                stmt = select(QARequest).where(QARequest.id == request_id)
                result = await session.execute(stmt)
                qa_request = result.scalar_one()
                
                qa_request.answer_text = answer
                qa_request.answer_model = model_name
                qa_request.confidence = confidence
                qa_request.is_refused = is_refused
                qa_request.latency_ms = int((time.time() - start_time) * 1000)
                
                # 获取检索配置（用于记录）
                retrieval_config = yaml_config.get("retrieval", {})
                rrf_k = retrieval_config.get("rrf_k", 60)
                
                qa_request.retrieval_json = {
                    'chunks_count': len(retrieved_chunks),
                    'chunk_ids': chunk_ids,
                    'extracted_chunk_ids': extracted_chunk_ids,
                    'multi_route_queries': len(rewritten_queries),
                    'rrf_fusion_enabled': True,
                    'rerank_enabled': retrieval_config.get("rerank_enabled", False),
                    'rrf_k': rrf_k
                }
                
                await session.commit()
            
            # 构建返回结果
            citations = []
            for chunk_id in extracted_chunk_ids:
                chunk = next((c for c in chunks_data if c['id'] == chunk_id), None)
                if chunk:
                    citations.append({
                        'chunk_id': chunk_id,
                        'content_preview': chunk['content'][:200] if chunk.get('content') else '',
                        'page_start': chunk.get('page_start'),
                        'page_end': chunk.get('page_end'),
                        'section_path': chunk.get('section_path')
                    })
            
            return {
                'success': True,
                'request_id': request_id,
                'answer': answer,
                'confidence': confidence,
                'is_refused': is_refused,
                'model_name': model_name,
                'citations': citations,
                'retrieved_chunks_count': len(retrieved_chunks),
                'cited_chunks_count': len(extracted_chunk_ids),
                'latency_ms': int((time.time() - start_time) * 1000)
            }
        
        except Exception as e:
            # 错误处理：更新请求状态
            if request_id:
                try:
                    async with db_manager.get_session() as session:
                        stmt = select(QARequest).where(QARequest.id == request_id)
                        result = await session.execute(stmt)
                        qa_request = result.scalar_one()
                        
                        qa_request.answer_text = f"处理失败: {str(e)}"
                        qa_request.is_refused = True
                        qa_request.latency_ms = int((time.time() - start_time) * 1000)
                        
                        await session.commit()
                except:
                    pass
            
            raise


# 创建全局单例
rag_service = RAGService()
