"""
RAG 核心服务
协调检索引擎和 LLM 服务，实现完整的 RAG 流程
"""
import asyncio
import hashlib
import json
import re
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.storage.db_manager import db_manager
from app.storage.models import QARequest, QACitation, Chunk
from app.llm import llm_service
from app.config.config import yaml_config
from app.pipeline.query import query_rewriter
from app.retrieval.rrf import merge_with_rrf
from app.router import intent_router
from app.tools.calculator import CalculatorTool

# 配置日志
import logging
logger = logging.getLogger(__name__)

# 幂等缓存（进程内）：{task_id: {"status": str, "result": dict}}
request_cache: Dict[str, Dict[str, Any]] = {}
request_cache_lock = asyncio.Lock()

STATUS_PENDING = "PENDING"
STATUS_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"

# CALC 意图：从参考资料中抽取数字并判断运算类型的系统提示
CALC_EXTRACT_SYSTEM_PROMPT = """你是一个数据分析助手。请阅读提供的【参考资料】，提取回答【用户问题】所需的数字，并判断运算类型。

输出格式要求：请仅输出一个 JSON 字符串，包含以下字段：
- op_type: "growth" | "diff" | "ratio" | "none"  （growth=增长率，diff=差值，ratio=占比；无法从资料中得到可计算数字时填 "none"）
- val1: 数字1（新值/分子/被减数）
- val2: 数字2（旧值/分母/减数）
- reason: 简短的理由

若无法从资料中提取出两个可参与计算的数字，则 op_type 填 "none"，val1/val2 可填 0。"""

# CALC 意图：根据计算结果与参考资料生成最终自然语言答案的系统提示
CALC_SYNTHESIS_SYSTEM_PROMPT = """你是一个问答助手。请根据【参考资料】和【计算结果】生成对用户问题的自然语言回答。

硬性要求：最终答案中必须包含计算公式（formula）的完整内容，不可省略。"""

# SUMMARY 意图：结构化总结报告的系统提示
SUMMARY_SYSTEM_PROMPT = """你是一名财务分析师。请根据【参考资料】生成结构化总结报告，用于回答用户的总结类问题。

报告须包含以下部分（可据资料适当精简）：
1. 基本情况
2. 财务指标
3. 亮点
4. 风险

硬性要求：每个关键结论或数据必须标注来源，使用 [Chunk ID] 格式引用，例如 [Chunk 123]。不得编造资料中未出现的信息。"""


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

    def _make_task_id(self, user_id: Optional[int], query: str) -> str:
        """基于 user_id + query 生成幂等 task_id（SHA256）"""
        uid = 0 if user_id is None else int(user_id)
        q = (query or "").strip()
        raw = f"{uid}|{q}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    async def query_idempotent(
        self,
        query: str,
        tenant_id: int = 0,
        user_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        带幂等性的 query：同一 user_id + query 在同一进程内保证去重执行。

        Returns:
            {
              "task_id": str,
              "status": "SUCCESS" | "PENDING" | "FAILED",
              "result": Optional[dict]
            }
        """
        task_id = self._make_task_id(user_id, query)

        async with request_cache_lock:
            entry = request_cache.get(task_id)
            if entry and entry.get("status") == STATUS_SUCCESS:
                logger.info(
                    "[Idempotency] Hit cache for task_id: %s, skipping execution.",
                    task_id,
                )
                return {"task_id": task_id, "status": STATUS_SUCCESS, "result": entry.get("result")}
            if entry and entry.get("status") == STATUS_PENDING:
                return {"task_id": task_id, "status": STATUS_PENDING, "result": None}

            # 任务锁定：先标记为 PENDING
            request_cache[task_id] = {"status": STATUS_PENDING, "result": None}

        try:
            result = await self.query(
                query=query,
                tenant_id=tenant_id,
                user_id=user_id,
                top_k=top_k,
            )
        except Exception as e:
            async with request_cache_lock:
                request_cache[task_id] = {
                    "status": STATUS_FAILED,
                    "result": {"success": False, "error": str(e)},
                }
            raise

        async with request_cache_lock:
            request_cache[task_id] = {"status": STATUS_SUCCESS, "result": result}

        return {"task_id": task_id, "status": STATUS_SUCCESS, "result": result}

    async def query_stream(
        self,
        query: str,
        tenant_id: int = 0,
        user_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式问答：先 yield 引用来源（citations），再逐块 yield 答案内容。
        若幂等缓存中已有 SUCCESS 结果，则直接 yield 该结果（先 citations 再 answer）；新请求走正常流式生成。
        """
        task_id = self._make_task_id(user_id, query)
        async with request_cache_lock:
            entry = request_cache.get(task_id)
            if entry and entry.get("status") == STATUS_SUCCESS:
                logger.info(
                    "[Idempotency] Hit cache for task_id: %s, skipping execution.",
                    task_id,
                )
                result = entry.get("result") or {}
                yield {"event": "citations", "data": result.get("citations", [])}
                yield {"event": "answer", "data": result.get("answer", "")}
                return
            if entry and entry.get("status") == STATUS_PENDING:
                yield {"event": "pending", "data": "请求正在处理中，请稍候"}
                return
            request_cache[task_id] = {"status": STATUS_PENDING, "result": None}

        start_time = time.time()
        request_id = None
        try:
            async with db_manager.get_session() as session:
                qa_request = await self.create_qa_request(
                    session, query, tenant_id, user_id
                )
                request_id = qa_request.id
                await session.commit()

            rewrite_result = await query_rewriter.rewrite_query(
                query=query, conversation_history=None, tenant_id=tenant_id
            )
            rewritten_queries = rewrite_result.get("rewritten_queries", [query])
            extracted_years = rewrite_result.get("extracted_years") or rewrite_result.get("rewrite_details", {}).get("extracted_years")

            retrieval_config = yaml_config.get("retrieval", {})
            multi_route_top_k = retrieval_config.get("multi_route_top_k", 10)
            rrf_k = retrieval_config.get("rrf_k", 60)
            top_k = top_k or self.default_top_k

            tasks = [
                self.retrieval_engine.retrieve(
                    query=q,
                    top_k=multi_route_top_k,
                    tenant_id=tenant_id,
                    years=extracted_years,
                )
                for q in rewritten_queries
            ]
            gather_results = await asyncio.gather(*tasks, return_exceptions=True)
            retrieval_results = []
            for i, q in enumerate(rewritten_queries):
                if isinstance(gather_results[i], Exception):
                    retrieval_results.append({"query": q, "chunks": []})
                else:
                    r = gather_results[i]
                    retrieval_results.append({"query": q, "chunks": r.get("chunks", [])})

            fused_chunks = merge_with_rrf(
                retrieval_results=retrieval_results,
                k=rrf_k,
                top_k=top_k * 2,
            )
            seen_ids = set()
            deduped = []
            for c in fused_chunks:
                cid = c.get("id")
                if cid is not None and cid not in seen_ids:
                    seen_ids.add(cid)
                    deduped.append(c)
            fused_chunks = deduped

            rerank_enabled = retrieval_config.get("rerank_enabled", False)
            rerank_top_k = retrieval_config.get("rerank_top_k", 5)
            retrieved_chunks = fused_chunks[:rerank_top_k] if rerank_enabled else fused_chunks[:top_k]

            if not retrieved_chunks:
                retrieved_chunks = []

            chunk_ids = [c["id"] for c in retrieved_chunks]
            async with db_manager.get_session() as session:
                chunks_data = await self.get_chunks_by_ids(
                    session, chunk_ids, tenant_id
                )
            context_text = self._build_context_from_chunks(chunks_data)
            intent = await intent_router.get_intent(query)

            citations = [
                {
                    "chunk_id": c["id"],
                    "content_preview": (c.get("content") or "")[:200],
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "section_path": c.get("section_path"),
                }
                for c in chunks_data
            ]
            # 引用只发送一次，不随每个 token 重复，减少 SSE 流量与前端解析压力
            yield {"event": "citations", "data": citations}

            answer = None
            model_name = "unknown"
            confidence = 0.0
            is_refused = False
            extracted_chunk_ids = list(chunk_ids)

            if intent == "CALC":
                calc_out = await self._handle_calc_logic(
                    query=query, context=context_text, request_id=request_id
                )
                if not calc_out.get("fallback"):
                    answer, model_name = await self._synthesize_calc_answer(
                        query=query,
                        context=context_text,
                        result=calc_out.get("result"),
                        formula=calc_out.get("formula", ""),
                        request_id=request_id,
                    )
                    confidence = 0.85
                    is_refused = False
            elif intent == "SUMMARY":
                answer, model_name = await self._handle_summary_logic(
                    query=query,
                    context=context_text,
                    request_id=request_id,
                )
                confidence = 0.9
                is_refused = False

            if answer is not None:
                for ch in answer:
                    yield {"event": "token", "data": ch}
                extracted_chunk_ids = chunk_ids
            else:
                full_answer = []
                async for token in llm_service.generate_answer_stream(
                    query=query,
                    context_chunks=chunks_data,
                    request_id=request_id,
                ):
                    full_answer.append(token)
                    yield {"event": "token", "data": token}
                answer = "".join(full_answer)
                extracted_chunk_ids = llm_service._extract_chunk_ids(answer)
                confidence = llm_service._calculate_confidence(answer, chunks_data)
                min_conf = yaml_config.get("qa_policy", {}).get("min_confidence", 0.6)
                is_refused = (
                    not answer
                    or (yaml_config.get("qa_policy", {}).get("refuse_if_low_conf", True) and confidence < min_conf)
                    or (yaml_config.get("qa_policy", {}).get("cite_required", True) and not extracted_chunk_ids)
                )

            # 零阻塞：流式结束后在后台写入 DB 与缓存，不阻塞 Token 发送
            asyncio.create_task(
                self._save_stream_result(
                    request_id=request_id,
                    answer=answer or "",
                    model_name=model_name,
                    confidence=confidence,
                    is_refused=is_refused,
                    extracted_chunk_ids=extracted_chunk_ids,
                    chunks_data=chunks_data,
                    retrieved_chunks=retrieved_chunks,
                    chunk_ids=chunk_ids,
                    task_id=task_id,
                    start_time=start_time,
                )
            )
        except Exception as e:
            async with request_cache_lock:
                request_cache[task_id] = {
                    "status": STATUS_FAILED,
                    "result": {"success": False, "error": str(e)},
                }
            yield {"event": "error", "data": str(e)}
            raise

    async def _save_stream_result(
        self,
        request_id: int,
        answer: str,
        model_name: str,
        confidence: float,
        is_refused: bool,
        extracted_chunk_ids: List[int],
        chunks_data: List[Dict[str, Any]],
        retrieved_chunks: List[Dict[str, Any]],
        chunk_ids: List[int],
        task_id: str,
        start_time: float,
    ) -> None:
        """流式结束后异步写入 DB 与缓存，不阻塞 Token 发送。"""
        try:
            async with db_manager.get_session() as session:
                await self.save_citations(
                    session, request_id, extracted_chunk_ids, chunks_data
                )
                stmt = select(QARequest).where(QARequest.id == request_id)
                res = await session.execute(stmt)
                qa_request = res.scalar_one()
                qa_request.answer_text = answer or ""
                qa_request.answer_model = model_name
                qa_request.confidence = confidence
                qa_request.is_refused = is_refused
                qa_request.latency_ms = int((time.time() - start_time) * 1000)
                qa_request.retrieval_json = {
                    "chunks_count": len(retrieved_chunks),
                    "chunk_ids": chunk_ids,
                    "extracted_chunk_ids": extracted_chunk_ids,
                }
                await session.commit()

            final_citations = [
                {
                    "chunk_id": c["id"],
                    "content_preview": (c.get("content") or "")[:200],
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "section_path": c.get("section_path"),
                }
                for c in chunks_data
                if c["id"] in extracted_chunk_ids
            ]
            final_result = {
                "success": True,
                "request_id": request_id,
                "answer": answer or "",
                "confidence": confidence,
                "is_refused": is_refused,
                "model_name": model_name,
                "citations": final_citations,
                "retrieved_chunks_count": len(retrieved_chunks),
                "cited_chunks_count": len(extracted_chunk_ids),
                "latency_ms": int((time.time() - start_time) * 1000),
            }
            async with request_cache_lock:
                request_cache[task_id] = {"status": STATUS_SUCCESS, "result": final_result}
        except Exception as e:
            logger.exception("流式后保存结果失败: %s", e)
            async with request_cache_lock:
                request_cache[task_id] = {
                    "status": STATUS_FAILED,
                    "result": {"success": False, "error": str(e)},
                }

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
                'table_id': chunk.table_id,
                'document_id': chunk.document_id,
            }
            for chunk in chunks
        ]

    @staticmethod
    def _merge_adjacent_chunks_by_document(
        chunks_data: List[Dict[str, Any]],
        page_window: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        将同一 document_id 且位置临近的 chunk 合并为一段上下文，保证连贯性。
        返回列表项为 {"chunk_ids": [id1, id2, ...], "content": "合并后的文本", "document_id": int}。
        """
        if not chunks_data:
            return []
        # 按 document_id、page_start 排序
        key = lambda c: (c.get("document_id") or 0, c.get("page_start") or 0)
        sorted_chunks = sorted(chunks_data, key=key)
        merged = []
        current_doc = None
        current_page_end = -1
        current_ids = []
        current_parts = []

        def flush():
            if current_ids and current_parts:
                merged.append({
                    "chunk_ids": current_ids[:],
                    "content": "\n\n".join(current_parts),
                    "document_id": current_doc,
                })

        for c in sorted_chunks:
            doc_id = c.get("document_id")
            page_start = c.get("page_start") or 0
            page_end = c.get("page_end") or page_start
            if doc_id != current_doc:
                flush()
                current_doc = doc_id
                current_page_end = page_end
                current_ids = [c["id"]]
                current_parts = [c.get("content", "")]
                continue
            if page_start <= current_page_end + page_window:
                current_ids.append(c["id"])
                current_parts.append(c.get("content", ""))
                current_page_end = max(current_page_end, page_end)
            else:
                flush()
                current_page_end = page_end
                current_ids = [c["id"]]
                current_parts = [c.get("content", "")]
        flush()
        return merged

    def _build_context_from_chunks(self, chunks_data: List[Dict[str, Any]]) -> str:
        """将 chunk 列表格式化为带 [Chunk ID] 的上下文字符串。若含 document_id 则先按同文档邻近合并再输出。"""
        if not chunks_data:
            return ""
        if any(c.get("document_id") is not None for c in chunks_data):
            merged = self._merge_adjacent_chunks_by_document(chunks_data)
            return "\n\n".join(
                " ".join(f"[Chunk {cid}]" for cid in group["chunk_ids"]) + "\n" + group["content"]
                for group in merged
            )
        return "\n\n".join(
            f"[Chunk {chunk['id']}]\n{chunk.get('content', '')}"
            for chunk in chunks_data
        )

    @staticmethod
    def _extract_json_from_llm(text: str) -> Optional[Dict[str, Any]]:
        """从 LLM 输出中提取 JSON（允许被 ```json ... ``` 包裹）"""
        if not text or not text.strip():
            return None
        text = text.strip()
        # 去掉 ```json ... ``` 或 ``` ... ```
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试直接整段解析
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None

    async def _handle_calc_logic(
        self,
        query: str,
        context: str,
        request_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        CALC 意图：从上下文中抽取数字、判断运算类型并执行计算。

        Returns:
            - fallback=True: 应退回到普通 RAG 回答（op_type 为 none 或解析/计算失败）
            - fallback=False: 已得到有效计算，含 result、formula、reason 等，供后续合成答案
        """
        user_message = f"""【参考资料】\n{context}\n\n【用户问题】\n{query}"""
        messages = [
            {"role": "system", "content": CALC_EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        try:
            response = await llm_service.chat_completion(
                messages=messages,
                stream=False,
                temperature=0.0,
                max_tokens=512,
                request_id=request_id,
            )
            raw = (
                response.choices[0].message.content
                if response.choices and response.choices[0].message
                else ""
            )
        except Exception as e:
            logger.warning("CALC 抽取 LLM 调用失败，退回到普通 RAG: %s", e)
            return {"fallback": True}

        data = self._extract_json_from_llm(raw)
        if not data:
            logger.warning("CALC 无法解析 LLM 输出的 JSON，退回普通 RAG。原始: %s", raw[:200])
            return {"fallback": True}

        op_type = (data.get("op_type") or "").strip().lower()
        if op_type == "none":
            return {"fallback": True}

        if op_type not in ("growth", "diff", "ratio"):
            logger.warning("CALC op_type 非法或为 none: %s，退回普通 RAG", op_type)
            return {"fallback": True}

        try:
            val1 = float(data.get("val1", 0))
            val2 = float(data.get("val2", 0))
        except (TypeError, ValueError):
            logger.warning("CALC val1/val2 无法转为数字，退回普通 RAG。data=%s", data)
            return {"fallback": True}

        calc = CalculatorTool().exec(op_type, val1, val2)
        return {
            "fallback": False,
            "op_type": op_type,
            "result": calc.get("result"),
            "formula": calc.get("formula", ""),
            "reason": data.get("reason", ""),
        }

    async def _synthesize_calc_answer(
        self,
        query: str,
        context: str,
        result: Any,
        formula: str,
        request_id: Optional[int] = None,
    ) -> tuple[str, str]:
        """根据计算结果与上下文，调用 LLM 生成包含公式的最终自然语言答案。返回 (answer, model_name)。"""
        result_str = str(result) if result is not None else "（计算未得到数值，见公式说明）"
        user_message = f"""【参考资料】\n{context}\n\n【计算结果】\nresult = {result_str}\nformula = {formula}\n\n【用户问题】\n{query}\n\n请生成回答，并在回答中完整包含上述 formula。"""
        messages = [
            {"role": "system", "content": CALC_SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        response = await llm_service.chat_completion(
            messages=messages,
            stream=False,
            temperature=0.3,
            max_tokens=1024,
            request_id=request_id,
        )
        answer = (
            response.choices[0].message.content
            if response.choices and response.choices[0].message
            else ""
        )
        model_name = getattr(response, "model", None) or "unknown"
        # 若模型未包含 formula，则强制追加
        if formula and formula.strip() and formula not in answer:
            answer = answer.rstrip() + "\n\n计算公式：" + formula
        return answer.strip(), model_name

    async def _handle_summary_logic(
        self,
        query: str,
        context: str,
        request_id: Optional[int] = None,
    ) -> tuple[str, str]:
        """SUMMARY 意图：基于上下文生成结构化总结报告。返回 (answer, model_name)。"""
        user_message = f"""【参考资料】\n{context}\n\n【用户问题】\n{query}"""
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        response = await llm_service.chat_completion(
            messages=messages,
            stream=False,
            temperature=0.3,
            max_tokens=2048,
            request_id=request_id,
        )
        answer = (
            response.choices[0].message.content
            if response.choices and response.choices[0].message
            else ""
        )
        model_name = getattr(response, "model", None) or "unknown"
        return answer.strip(), model_name
    
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
            
            # 3. 多路召回：使用改写后的查询并发检索（每路携带时间约束）
            retrieval_config = yaml_config.get("retrieval", {})
            multi_route_top_k = retrieval_config.get("multi_route_top_k", 10)  # 每个查询取Top 10
            rrf_k = retrieval_config.get("rrf_k", 60)  # RRF常数k
            extracted_years = rewrite_result.get("extracted_years") or rewrite_result.get("rewrite_details", {}).get("extracted_years")
            if extracted_years:
                logger.info("[Year Filter] 多路检索携带时间约束: years=%s", extracted_years)

            retrieval_start = time.time()
            tasks = [
                self.retrieval_engine.retrieve(
                    query=rewritten_query,
                    top_k=multi_route_top_k,
                    tenant_id=tenant_id,
                    years=extracted_years,
                )
                for rewritten_query in rewritten_queries
            ]
            gather_results = await asyncio.gather(*tasks, return_exceptions=True)
            retrieval_elapsed = time.time() - retrieval_start
            logger.info("并发检索耗时: %.2fs", retrieval_elapsed)

            retrieval_results = []
            for i, rewritten_query in enumerate(rewritten_queries):
                if isinstance(gather_results[i], Exception):
                    logger.warning(
                        "[Multi-Route Retrieval] 查询 '%s...' 检索异常: %s",
                        rewritten_query[:30],
                        gather_results[i],
                    )
                    retrieval_results.append({"query": rewritten_query, "chunks": []})
                else:
                    retrieval_result = gather_results[i]
                    chunks = retrieval_result.get("chunks", [])
                    retrieval_results.append({"query": rewritten_query, "chunks": chunks})
                    logger.info(
                        "[Multi-Route Retrieval] 查询 '%s...' 检索到 %d 个结果",
                        rewritten_query[:30],
                        len(chunks),
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

            # 4.5 按 chunk_id 严格去重，保留得分最高的一条
            seen_ids = set()
            deduped_chunks = []
            for c in fused_chunks:
                cid = c.get("id")
                if cid is not None and cid not in seen_ids:
                    seen_ids.add(cid)
                    deduped_chunks.append(c)
            if len(deduped_chunks) < len(fused_chunks):
                logger.info(
                    "[Dedup] 按 chunk_id 去重: %d -> %d",
                    len(fused_chunks), len(deduped_chunks),
                )
            fused_chunks = deduped_chunks

            logger.info(
                "[RRF Fusion] 融合完成，得到 %d 个候选chunks", len(fused_chunks)
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
            context_text = self._build_context_from_chunks(chunks_data)
            
            # 6. 意图识别：CALC 时走计算+合成，否则走普通 RAG 生成
            intent = await intent_router.get_intent(query)
            answer = None
            extracted_chunk_ids = list(chunk_ids)

            if intent == "CALC":
                calc_out = await self._handle_calc_logic(
                    query=query,
                    context=context_text,
                    request_id=request_id,
                )
                if not calc_out.get("fallback"):
                    # 执行计算并合成最终答案（答案中必须包含 formula）
                    answer, model_name = await self._synthesize_calc_answer(
                        query=query,
                        context=context_text,
                        result=calc_out.get("result"),
                        formula=calc_out.get("formula", ""),
                        request_id=request_id,
                    )
                    confidence = 0.85
                    is_refused = False
            elif intent == "SUMMARY":
                answer, model_name = await self._handle_summary_logic(
                    query=query,
                    context=context_text,
                    request_id=request_id,
                )
                confidence = 0.9
                is_refused = False

            if answer is None:
                # 普通 RAG 或 CALC 退回到普通 RAG
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
