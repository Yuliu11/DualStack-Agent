"""
文档摄取流水线
整合解析、清洗、切分、增量更新和持久化
"""
import os
import re
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from pathlib import Path

from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.mysql import insert

from app.pipeline.parser import PDFParser, ParsedPDF, TextElement, TableElement
from app.pipeline.cleaner import Cleaner
from app.pipeline.chunker import DocumentChunker
from app.storage.db_manager import db_manager
from app.storage.models import Document, DocVersion, Chunk, Table, Embedding, IndexBuild
from app.config.config import yaml_config, settings
from app.retrieval.vector_store import VectorStore
from openai import AsyncOpenAI
import json

# 配置日志
logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """文档摄取流水线"""
    
    def __init__(self):
        """初始化流水线"""
        self.parser = PDFParser()
        self.cleaner = Cleaner()
        self.chunker = DocumentChunker(
            chunk_size=500,
            chunk_overlap=100
        )
        self.default_tenant_id = 0
        
        # 初始化向量存储（解析/摄取模式：允许写入索引）
        vector_config = yaml_config.get("vector_store", {})
        vector_store_type = vector_config.get("default", "faiss")
        index_path = vector_config.get("faiss", {}).get("index_path", "./data/faiss_index")
        self.vector_store = VectorStore(
            vector_store_type=vector_store_type,
            index_path=index_path,
            read_only=False,
        )
    
    def calculate_content_hash(self, content: str) -> str:
        """计算内容的 SHA256 哈希值"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def find_existing_chunk(
        self,
        session: AsyncSession,
        tenant_id: int,
        content_sha256: str
    ) -> Optional[Chunk]:
        """查找已存在的 chunk（用于增量更新）"""
        stmt = select(Chunk).where(
            and_(
                Chunk.tenant_id == tenant_id,
                Chunk.content_sha256 == content_sha256,
                Chunk.is_deleted == False
            )
        ).limit(1)
        
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def find_existing_embedding(
        self,
        session: AsyncSession,
        tenant_id: int,
        embed_sha256: str,
        embed_model: str
    ) -> Optional[Embedding]:
        """
        查找已存在的 embedding（用于向量复用）
        
        Args:
            session: 数据库会话
            tenant_id: 租户ID
            embed_sha256: 嵌入内容的 SHA256 哈希（通常等于 content_sha256）
            embed_model: 嵌入模型名称
        
        Returns:
            已存在的 Embedding 对象，如果不存在则返回 None
        """
        stmt = select(Embedding).where(
            and_(
                Embedding.tenant_id == tenant_id,
                Embedding.embed_sha256 == embed_sha256,
                Embedding.embed_model == embed_model,
                Embedding.status == "active"
            )
        ).limit(1)
        
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def find_existing_table(
        self,
        session: AsyncSession,
        tenant_id: int,
        table_sha256: str
    ) -> Optional[Table]:
        """查找已存在的表格（用于增量更新）"""
        stmt = select(Table).where(
            and_(
                Table.tenant_id == tenant_id,
                Table.table_sha256 == table_sha256
            )
        ).limit(1)
        
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def check_document_exists(
        self,
        session: AsyncSession,
        tenant_id: int,
        file_sha256: str
    ) -> Optional[Document]:
        """
        两级校验：第一级 - 检查文档级 file_sha256 是否已存在
        
        Returns:
            已存在的文档对象，如果不存在则返回 None
        """
        stmt = select(Document).where(
            and_(
                Document.tenant_id == tenant_id,
                Document.file_sha256 == file_sha256
            )
        ).limit(1)
        
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def find_document_by_filename(
        self,
        session: AsyncSession,
        tenant_id: int,
        filename: str
    ) -> Optional[Document]:
        """
        根据文件名查找文档（从 file_uri 中提取文件名进行匹配）
        
        Args:
            session: 数据库会话
            tenant_id: 租户ID
            filename: 文件名（不含路径）
        
        Returns:
            如果存在返回 Document 对象，否则返回 None
        """
        # 使用 LIKE 查询匹配 file_uri 以文件名结尾的记录
        # 这样可以匹配不同路径但文件名相同的文件
        stmt = select(Document).where(
            and_(
                Document.tenant_id == tenant_id,
                Document.file_uri.like(f"%/{filename}") | Document.file_uri.like(f"%\\{filename}")
            )
        ).limit(1)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def delete_document_and_related(
        self,
        session: AsyncSession,
        document_id: int
    ) -> bool:
        """
        删除文档及其所有相关记录（手动级联删除）
        
        删除顺序：
        1. qa_citations（引用 chunks 的记录）
        2. embeddings（通过 chunks）
        3. chunks
        4. tables
        5. doc_versions
        6. documents（最后删除）
        
        Args:
            session: 数据库会话
            document_id: 文档ID
        
        Returns:
            是否成功删除
        """
        try:
            from app.storage.models import Embedding, Chunk, DocVersion, Table, QACitation
            
            # 1. 先获取所有相关的 chunk_ids
            chunks_stmt = select(Chunk.id).where(Chunk.document_id == document_id)
            chunks_result = await session.execute(chunks_stmt)
            chunk_ids = [row[0] for row in chunks_result.fetchall()]
            
            # 2. 删除 qa_citations（必须先删除，因为外键约束）
            if chunk_ids:
                citations_delete_stmt = delete(QACitation).where(
                    QACitation.chunk_id.in_(chunk_ids)
                )
                await session.execute(citations_delete_stmt)
                logger.debug(f"已删除 {len(chunk_ids)} 个 chunks 的引用记录")
            
            # 3. 删除 embeddings（通过 chunks）
            if chunk_ids:
                embeddings_delete_stmt = delete(Embedding).where(
                    Embedding.chunk_id.in_(chunk_ids)
                )
                await session.execute(embeddings_delete_stmt)
            
            # 4. 删除 chunks
            chunks_delete_stmt = delete(Chunk).where(Chunk.document_id == document_id)
            await session.execute(chunks_delete_stmt)
            
            # 5. 删除 tables
            tables_delete_stmt = delete(Table).where(Table.document_id == document_id)
            await session.execute(tables_delete_stmt)
            
            # 6. 删除 doc_versions
            versions_delete_stmt = delete(DocVersion).where(DocVersion.document_id == document_id)
            await session.execute(versions_delete_stmt)
            
            # 7. 最后删除文档本身
            doc_delete_stmt = delete(Document).where(Document.id == document_id)
            await session.execute(doc_delete_stmt)
            
            # 提交所有删除操作
            await session.flush()
            
            logger.info(f"成功删除文档及相关记录: document_id={document_id}")
            return True
        except Exception as e:
            logger.error(f"删除文档及相关记录失败: document_id={document_id}, 错误: {e}", exc_info=True)
            await session.rollback()
            return False
    
    async def create_document(
        self,
        session: AsyncSession,
        file_uri: str,
        file_sha256: str,
        file_size: int,
        title: str,
        tenant_id: int = 0,
        doc_type: str = "annual_report",
        source: str = "upload",
        company_name: Optional[str] = None,
        stock_code: Optional[str] = None,
        year: Optional[int] = None,
        report_date: Optional[date] = None
    ) -> Document:
        """创建文档记录"""
        # 检查是否已存在
        existing_doc = await self.check_document_exists(session, tenant_id, file_sha256)
        
        if existing_doc:
            # 更新状态为解析中
            existing_doc.parse_status = "pending"
            existing_doc.parse_error = None
            logger.info(f"文档已存在 (file_sha256={file_sha256[:16]}...), 跳过文档级处理")
            return existing_doc
        
        # 创建新文档
        document = Document(
            tenant_id=tenant_id,
            doc_type=doc_type,
            source=source,
            title=title,
            company_name=company_name,
            stock_code=stock_code,
            year=year,
            report_date=report_date,
            file_uri=file_uri,
            file_sha256=file_sha256,
            file_size=file_size,
            parse_status="pending"  # 初始状态为 pending
        )
        session.add(document)
        await session.flush()  # 获取 ID
        
        logger.info(f"创建新文档 (file_sha256={file_sha256[:16]}...)")
        return document
    
    async def create_document_version(
        self,
        session: AsyncSession,
        document_id: int,
        version_no: int,
        content_sha256: str,
        tenant_id: int = 0,
        version_tag: Optional[str] = None
    ) -> DocVersion:
        """创建文档版本"""
        version = DocVersion(
            tenant_id=tenant_id,
            document_id=document_id,
            version_no=version_no,
            version_tag=version_tag,
            content_sha256=content_sha256,
            status="active"
        )
        session.add(version)
        await session.flush()  # 获取 ID
        return version
    
    async def process_table(
        self,
        session: AsyncSession,
        table_data: Dict[str, Any],
        document_id: int,
        version_id: int,
        page_no: int,
        tenant_id: int = 0
    ) -> Tuple[Table, str]:
        """
        处理表格，保存到 tables 表并返回 Markdown 格式
        
        Returns:
            (Table对象, Markdown格式的表格内容)
        """
        # 计算表格哈希
        table_json = json.dumps(table_data, ensure_ascii=False, sort_keys=True)
        table_sha256 = self.calculate_content_hash(table_json)
        
        # 检查是否已存在
        existing_table = await self.find_existing_table(session, tenant_id, table_sha256)
        if existing_table:
            # 复用已存在的表格
            markdown_content = self.parser.table_to_markdown(table_data)
            return existing_table, markdown_content
        
        # 创建新表格
        table = Table(
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
            page_no=page_no,
            table_title=None,
            schema_json=table_data.get("header"),
            data_json=table_data,
            table_sha256=table_sha256
        )
        session.add(table)
        await session.flush()  # 获取 ID
        
        # 转换为 Markdown
        markdown_content = self.parser.table_to_markdown(table_data)
        
        return table, markdown_content
    
    async def process_chunks_batch(
        self,
        session: AsyncSession,
        chunks_data: List[Dict[str, Any]],
        document_id: int,
        version_id: int,
        tenant_id: int = 0,
        embed_model: str = "text-embedding-v2"
    ) -> Tuple[List[Chunk], int, int, int]:
        """
        批量处理 chunks，实现增量更新和向量复用
        
        Args:
            session: 数据库会话
            chunks_data: Chunk 数据列表
            document_id: 文档ID
            version_id: 版本ID
            tenant_id: 租户ID
            embed_model: 嵌入模型名称（用于向量复用查询）
        
        Returns:
            (创建的 Chunk 对象列表, 复用的 chunk 数量, 新增的 chunk 数量, 可复用向量的数量)
        """
        created_chunks = []
        reused_chunk_count = 0
        new_chunk_count = 0
        reusable_embedding_count = 0
        
        for idx, chunk_data in enumerate(chunks_data, start=1):
            content = chunk_data['content']
            content_sha256 = self.calculate_content_hash(content)
            
            # 两级校验：第二级 - 检查 Chunk 级 content_sha256 是否已存在
            existing_chunk = await self.find_existing_chunk(
                session,
                tenant_id,
                content_sha256
            )
            
            if existing_chunk:
                # Chunk 已存在，标记为复用
                reused_chunk_count += 1
                
                # 检查是否可以复用向量（在写入 embeddings 表前）
                # embed_sha256 通常等于 content_sha256（或 content_sha256 + model_name）
                embed_sha256 = content_sha256  # 简化：使用 content_sha256
                
                existing_embedding = await self.find_existing_embedding(
                    session,
                    tenant_id,
                    embed_sha256,
                    embed_model
                )
                
                if existing_embedding:
                    # 可以复用向量，记录 vector_id 信息
                    reusable_embedding_count += 1
                    logger.debug(
                        f"Chunk {idx}: 复用已存在的向量 "
                        f"(embed_sha256={embed_sha256[:16]}..., "
                        f"vector_id={existing_embedding.vector_id})"
                    )
            else:
                # 新 Chunk
                new_chunk_count += 1
            
            # 创建新 chunk（即使是复用，也创建新记录以保持版本独立性）
            chunk = Chunk(
                tenant_id=tenant_id,
                document_id=document_id,
                version_id=version_id,
                chunk_no=idx,
                page_start=chunk_data.get('page_no'),
                page_end=chunk_data.get('page_no'),
                section_path=None,  # 可以从标题结构提取
                content=content,
                content_sha256=content_sha256,
                content_tokens=None,
                is_table=chunk_data.get('is_table', False),
                table_id=chunk_data.get('table_id'),
                is_deleted=False
            )
            session.add(chunk)
            created_chunks.append(chunk)
        
        await session.flush()  # 获取所有 ID
        
        # 日志审计
        logger.info(
            f"本次处理新增 {new_chunk_count} 个 Chunk，"
            f"复用 {reused_chunk_count} 个已存在 Chunk，"
            f"其中 {reusable_embedding_count} 个可复用向量"
        )
        
        return created_chunks, reused_chunk_count, new_chunk_count, reusable_embedding_count
    
    async def _generate_embedding(
        self,
        text: str,
        embed_model: str
    ) -> Optional[List[float]]:
        """
        生成文本的嵌入向量
        
        Args:
            text: 文本内容
            embed_model: 嵌入模型名称
        
        Returns:
            嵌入向量，如果失败返回None
        """
        try:
            # 创建DashScope客户端（使用OpenAI兼容接口）
            dashscope_base_url = settings.dashscope_base_url
            
            # 确保使用兼容模式路径
            if "/compatible-mode" not in dashscope_base_url:
                if dashscope_base_url.endswith("/v1"):
                    dashscope_base_url = dashscope_base_url.replace("/v1", "/compatible-mode/v1")
                elif not dashscope_base_url.endswith("/compatible-mode/v1"):
                    dashscope_base_url = f"{dashscope_base_url.rstrip('/')}/compatible-mode/v1"
            
            dashscope_client = AsyncOpenAI(
                api_key=settings.dashscope_api_key,
                base_url=dashscope_base_url,
                timeout=30.0  # 索引构建时可以设置更长的超时
            )
            
            # 调用embedding API
            response = await dashscope_client.embeddings.create(
                model=embed_model,
                input=text
            )
            
            if response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                return embedding
            else:
                logger.warning(f"Empty embedding response for text: {text[:50]}...")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def build_index_for_chunks(
        self,
        session: AsyncSession,
        chunks: List[Chunk],
        embed_model: str,
        tenant_id: int = 0,
        document_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        为 chunks 构建向量索引
        
        Args:
            session: 数据库会话
            chunks: Chunk 对象列表
            embed_model: 嵌入模型名称
            tenant_id: 租户ID
            document_id: 文档ID（可选，用于记录）
        
        Returns:
            构建结果统计
        """
        if not chunks:
            return {
                'success': True,
                'processed': 0,
                'failed': 0,
                'skipped': 0
            }
        
        processed_count = 0
        failed_count = 0
        skipped_count = 0
        vectors_to_add = []
        chunk_ids_to_add = []
        embeddings_to_save = []
        
        # 批量生成向量
        for chunk in chunks:
            # 检查是否已存在 embedding
            existing_embedding = await self.find_existing_embedding(
                session,
                tenant_id,
                chunk.content_sha256,
                embed_model
            )
            
            if existing_embedding:
                # 已存在，跳过生成，但需要添加到索引（如果索引中没有）
                skipped_count += 1
                logger.debug(f"Chunk {chunk.id}: 复用已存在的向量")
                
                # 检查索引中是否已有该向量
                vector_id = existing_embedding.vector_id
                if self.vector_store.vector_store_type == "faiss":
                    # 对于 FAISS，需要检查映射中是否已有
                    if vector_id not in self.vector_store.chunk_to_id:
                        # 需要从数据库加载向量并添加到索引
                        # 这里简化处理：跳过已存在的向量
                        continue
                continue
            
            # 生成新向量（失败时跳过该 chunk，不中断整条流水线）
            embedding = await self._generate_embedding(chunk.content, embed_model)
            if embedding is None:
                failed_count += 1
                logger.error("Chunk %s: _generate_embedding 返回 None，跳过该 chunk", chunk.id)
                continue
            
            # 准备添加到索引
            vectors_to_add.append(embedding)
            chunk_ids_to_add.append(chunk.id)
            
            # 准备保存到数据库
            vector_id = f"faiss_{chunk.id}" if self.vector_store.vector_store_type == "faiss" else f"qdrant_{chunk.id}"
            embed_dim = len(embedding)
            
            embedding_record = Embedding(
                tenant_id=tenant_id,
                chunk_id=chunk.id,
                embed_model=embed_model,
                embed_dim=embed_dim,
                vector_store=self.vector_store.vector_store_type,
                vector_id=vector_id,
                embed_sha256=chunk.content_sha256,
                status="active"
            )
            embeddings_to_save.append(embedding_record)
            processed_count += 1
        
        # 批量添加到向量索引（先添加到索引，再统一保存一次）
        if vectors_to_add:
            try:
                self.vector_store.add_vectors(
                    vectors=vectors_to_add,
                    chunk_ids=chunk_ids_to_add,
                    vector_ids=[f"faiss_{cid}" if self.vector_store.vector_store_type == "faiss" else f"qdrant_{cid}"
                               for cid in chunk_ids_to_add]
                )
                logger.info("已添加 %d 个向量到索引", len(vectors_to_add))
            except Exception as e:
                logger.error(f"添加向量到索引失败: {e}", exc_info=True)
                # 索引添加失败，标记这些向量为失败
                failed_count += len(vectors_to_add)
                # 从待保存列表中移除失败的向量
                embeddings_to_save = [
                    emb for i, emb in enumerate(embeddings_to_save)
                    if i < len(embeddings_to_save) - len(vectors_to_add)
                ]
                vectors_to_add = []
                chunk_ids_to_add = []
        
        # 批量保存到数据库（只保存成功添加到索引的向量元数据）
        if embeddings_to_save:
            try:
                for emb in embeddings_to_save:
                    session.add(emb)
                await session.flush()
                logger.info("已保存 %d 个向量元数据到数据库", len(embeddings_to_save))
            except Exception as e:
                logger.error("保存向量元数据失败: %s", e, exc_info=True)
                await session.rollback()
                return {
                    'success': False,
                    'processed': processed_count - len(embeddings_to_save),
                    'failed': failed_count + len(embeddings_to_save),
                    'skipped': skipped_count,
                    'error': str(e)
                }

        # 仅在所有向量成功添加后执行一次索引落盘
        if vectors_to_add and self.vector_store.vector_store_type == "faiss":
            try:
                self.vector_store.save_index()
                logger.info("索引已保存到文件")
            except Exception as save_err:
                logger.error("保存索引失败: %s", save_err, exc_info=True)

        return {
            'success': True,
            'processed': processed_count,
            'failed': failed_count,
            'skipped': skipped_count
        }
    
    async def ingest_pdf(
        self,
        file_path: str,
        tenant_id: int = 0,
        doc_type: str = "annual_report",
        source: str = "upload",
        title: Optional[str] = None,
        company_name: Optional[str] = None,
        stock_code: Optional[str] = None,
        year: Optional[int] = None,
        report_date: Optional[date] = None,
        version_tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        摄取 PDF 文档的主流程。幂等：相同 file_sha256 且已解析则直接返回，不调用解析。
        分三阶段提交：创建/获取文档 -> 生成切片 -> 构建索引。
        """
        document = None
        file_sha256 = None

        try:
            # ---------- 强制前置校验：在进入任何 pdf_parser 解析逻辑之前，先计算 file_sha256 并查库 ----------
            file_sha256 = self.parser.calculate_file_hash(file_path)
            logger.info("开始处理文档 (file_sha256=%s...)", file_sha256[:16])

            async with db_manager.get_session() as session:
                existing_doc = await self.check_document_exists(session, tenant_id, file_sha256)
                if existing_doc and getattr(existing_doc, "parse_status", None) == "parsed":
                    logger.info(
                        "[Ingest] 文档已存在且已解析，跳过处理: file_sha256=%s..., document_id=%s",
                        file_sha256[:16], existing_doc.id,
                    )
                    return {
                        "success": True,
                        "document_id": existing_doc.id,
                        "message": "文档已存在且已解析，跳过处理",
                        "file_sha256": file_sha256,
                        "chunks_count": 0,
                        "reused_chunks_count": 0,
                        "new_chunks_count": 0,
                        "reusable_embeddings_count": 0,
                    }

            # ---------- 阶段一：创建/获取文档 ----------
            file_size = 0
            try:
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            except Exception:
                pass
            title_initial = title or Path(file_path).stem

            async with db_manager.get_session() as session:
                document = await self.create_document(
                    session,
                    file_uri=file_path,
                    file_sha256=file_sha256,
                    file_size=file_size,
                    title=title_initial,
                    tenant_id=tenant_id,
                    doc_type=doc_type,
                    source=source,
                    company_name=company_name,
                    stock_code=stock_code,
                    year=year,
                    report_date=report_date,
                )
                await session.commit()

            # ---------- 阶段二：解析、生成切片并持久化 ----------
            parsed_pdf = await self.parser.parse_pdf(file_path, extract_tables=True)
            if parsed_pdf.file_sha256 != file_sha256:
                raise ValueError(
                    "文件哈希不一致: 计算值=%s, 解析值=%s"
                    % (file_sha256, parsed_pdf.file_sha256)
                )
            title = title or parsed_pdf.title or Path(file_path).stem

            all_elements = []
            all_tables = []
            for page_elements in parsed_pdf.pages:
                # 清洗文本元素
                cleaned_text_elements = []
                for text_elem in page_elements.text_elements:
                    cleaned_text = self.cleaner.clean_text(
                        text_elem.text,
                        preserve_titles=True
                    )
                    if cleaned_text:
                        cleaned_text_elements.append(TextElement(
                            page_no=text_elem.page_no,
                            text=cleaned_text,
                            element_type=text_elem.element_type,
                            bbox=text_elem.bbox
                        ))
                
                # 合并文本和表格元素
                page_mixed = self.parser.merge_text_and_tables(
                    cleaned_text_elements,
                    page_elements.table_elements,
                    page_elements.page_no
                )
                all_elements.extend(page_mixed)
                
                # 收集表格数据
                for table_elem in page_elements.table_elements:
                    all_tables.append({
                        'data': table_elem.table_data,
                        'page_no': table_elem.page_no
                    })
            
            # 5. 切分（段落优先，使用 RecursiveCharacterTextSplitter 逻辑）
            chunks_data = self.chunker.split_elements(all_elements)
            
            # 6. 持久化到数据库（阶段二）
            async with db_manager.get_session() as session:
                doc_id = document.id
                # 确定版本号
                stmt = select(DocVersion.version_no).where(
                    DocVersion.document_id == doc_id
                ).order_by(DocVersion.version_no.desc()).limit(1)
                result = await session.execute(stmt)
                max_version = result.scalar_one_or_none()
                version_no = (max_version or 0) + 1

                all_content = "\n".join([c["content"] for c in chunks_data])
                content_sha256 = self.calculate_content_hash(all_content)

                version = await self.create_document_version(
                    session,
                    document_id=doc_id,
                    version_no=version_no,
                    content_sha256=content_sha256,
                    tenant_id=tenant_id,
                    version_tag=version_tag,
                )

                table_id_map = {}
                for table_info in all_tables:
                    table_obj, markdown_content = await self.process_table(
                        session,
                        table_info["data"],
                        document_id=doc_id,
                        version_id=version.id,
                        page_no=table_info["page_no"],
                        tenant_id=tenant_id,
                    )
                    table_id_map[table_obj.id] = markdown_content

                for chunk_data in chunks_data:
                    if chunk_data.get("is_table") and chunk_data.get("table_data"):
                        for table_id in table_id_map:
                            chunk_data["table_id"] = table_id
                            break

                embed_model = yaml_config.get("models", {}).get("default_embedding", "text-embedding-v2")
                chunks, reused_chunk_count, new_chunk_count, reusable_embedding_count = await self.process_chunks_batch(
                    session,
                    chunks_data,
                    document_id=doc_id,
                    version_id=version.id,
                    tenant_id=tenant_id,
                    embed_model=embed_model,
                )

                # 在本 session 内重新加载并更新文档状态
                doc_stmt = select(Document).where(Document.id == doc_id).limit(1)
                doc_result = await session.execute(doc_stmt)
                doc_row = doc_result.scalar_one_or_none()
                if doc_row:
                    doc_row.parse_status = "parsed"
                    doc_row.parse_error = None
                    doc_row.current_version_id = version.id

                await session.commit()
            # 阶段二结束（一次 commit）

            # ---------- 阶段三：构建向量索引（独立 session，一次 commit） ----------
            embed_model = yaml_config.get("models", {}).get("default_embedding", "text-embedding-v2")
            index_build_result = None
            async with db_manager.get_session() as session:
                # 重新加载 chunks（新 session）
                stmt = select(Chunk).where(
                    and_(
                        Chunk.document_id == document.id,
                        Chunk.is_deleted == False,
                    )
                )
                result = await session.execute(stmt)
                chunks = list(result.scalars().all())
                try:
                    index_build_result = await self.build_index_for_chunks(
                        session,
                        chunks,
                        embed_model,
                        tenant_id,
                        document.id,
                    )
                    await session.commit()
                    logger.info(
                        "索引构建完成: 处理 %s 个, 跳过 %s 个, 失败 %s 个",
                        index_build_result.get("processed", 0),
                        index_build_result.get("skipped", 0),
                        index_build_result.get("failed", 0),
                    )
                except Exception as e:
                    logger.error("索引构建失败: %s", e, exc_info=True)
                    await session.rollback()
                    index_build_result = {"success": False, "error": str(e)}
                    try:
                        if self.vector_store.vector_store_type == "faiss":
                            self.vector_store.save_index()
                            logger.info("已保存部分索引到文件")
                    except Exception as save_error:
                        logger.error("保存部分索引失败: %s", save_error)

            return {
                "success": True,
                "document_id": doc_id,
                "version_id": version.id,
                "version_no": version_no,
                "chunks_count": len(chunks),
                "reused_chunks_count": reused_chunk_count,
                "new_chunks_count": new_chunk_count,
                "reusable_embeddings_count": reusable_embedding_count,
                "tables_count": len(all_tables),
                "file_sha256": parsed_pdf.file_sha256,
                "index_build": index_build_result,
            }
        
        except Exception as e:
            # 异常捕获：删除失败的文档记录，避免残留
            error_msg = str(e)
            document_id_to_delete = None
            
            try:
                async with db_manager.get_session() as session:
                    # 查找文档（如果已创建）
                    doc_to_delete = None
                    if document:
                        stmt = select(Document).where(
                            Document.id == document.id
                        ).limit(1)
                        result = await session.execute(stmt)
                        doc_to_delete = result.scalar_one_or_none()
                    elif file_sha256:
                        stmt = select(Document).where(
                            and_(
                                Document.tenant_id == tenant_id,
                                Document.file_sha256 == file_sha256
                            )
                        ).limit(1)
                        result = await session.execute(stmt)
                        doc_to_delete = result.scalar_one_or_none()
                    else:
                        # 尝试计算文件哈希
                        try:
                            file_sha256 = self.parser.calculate_file_hash(file_path)
                            stmt = select(Document).where(
                                and_(
                                    Document.tenant_id == tenant_id,
                                    Document.file_sha256 == file_sha256
                                )
                            ).limit(1)
                            result = await session.execute(stmt)
                            doc_to_delete = result.scalar_one_or_none()
                        except:
                            pass
                    
                    if doc_to_delete:
                        document_id_to_delete = doc_to_delete.id
                        # 使用 delete_document_and_related 完整删除文档及其所有相关记录
                        deleted = await self.delete_document_and_related(
                            session, document_id_to_delete
                        )
                        if deleted:
                            await session.commit()
                            logger.warning(
                                f"删除失败的文档记录: document_id={document_id_to_delete}, "
                                f"file_sha256={file_sha256[:16] if file_sha256 else 'N/A'}..., "
                                f"错误: {error_msg[:100]}"
                            )
                        else:
                            await session.rollback()
                            logger.error(
                                f"删除失败文档记录时出错: document_id={document_id_to_delete}"
                            )
            except Exception as delete_error:
                # 删除失败不影响错误返回，但记录日志
                logger.error(f"删除失败文档记录时出错: {delete_error}", exc_info=True)
            
            return {
                'success': False,
                'error': error_msg,
                'document_id': document_id_to_delete
            }
    
    async def process_local_folder(
        self,
        folder_path: str,
        tenant_id: int = 0,
        doc_type: str = "annual_report",
        source: str = "local_scan"
    ) -> Dict[str, Any]:
        """
        批量处理本地文件夹中的 PDF 文件
        
        Args:
            folder_path: 文件夹路径
            tenant_id: 租户ID
            doc_type: 文档类型
            source: 来源标识
        
        Returns:
            处理结果统计
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.warning(f"文件夹不存在: {folder_path}")
            return {
                'success': False,
                'error': f"文件夹不存在: {folder_path}",
                'processed': 0,
                'skipped': 0,
                'failed': 0
            }
        
        if not folder.is_dir():
            logger.warning(f"路径不是文件夹: {folder_path}")
            return {
                'success': False,
                'error': f"路径不是文件夹: {folder_path}",
                'processed': 0,
                'skipped': 0,
                'failed': 0
            }
        
        # 查找所有 PDF 文件
        pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
        
        if not pdf_files:
            logger.info(f"文件夹中没有找到 PDF 文件: {folder_path}")
            return {
                'success': True,
                'message': "文件夹中没有找到 PDF 文件",
                'processed': 0,
                'skipped': 0,
                'failed': 0
            }
        
        logger.info(f"开始扫描文件夹: {folder_path}, 找到 {len(pdf_files)} 个 PDF 文件")
        
        processed_count = 0
        updated_count = 0  # 覆盖更新的文件数
        skipped_count = 0
        failed_count = 0
        results = []
        
        for pdf_file in pdf_files:
            existing_doc_by_name = None  # 用于跟踪是否是覆盖更新
            try:
                file_path = str(pdf_file)
                file_name = pdf_file.name
                logger.info(f"处理文件: {file_name}")
                
                # 覆盖式更新：先根据文件名查找并删除旧记录
                async with db_manager.get_session() as session:
                    existing_doc_by_name = await self.find_document_by_filename(
                        session, tenant_id, file_name
                    )
                    
                    if existing_doc_by_name:
                        logger.info(
                            f"发现同名文件，执行覆盖更新: {file_name} "
                            f"(document_id={existing_doc_by_name.id})"
                        )
                        
                        # 删除旧记录及其所有相关数据
                        deleted = await self.delete_document_and_related(
                            session, existing_doc_by_name.id
                        )
                        
                        doc_id = existing_doc_by_name.id
                        if deleted:
                            await session.commit()
                            logger.info(f"已删除旧记录: {file_name} (document_id={doc_id})")
                        else:
                            await session.rollback()
                            logger.warning(f"删除旧记录失败: {file_name} (document_id={doc_id})")
                            # 继续处理，让 ingest_pdf 处理冲突
                
                # 处理文件（重新解析和入库）
                result = await self.ingest_pdf(
                    file_path=file_path,
                    tenant_id=tenant_id,
                    doc_type=doc_type,
                    source=source,
                    title=None,  # 自动提取
                    company_name=None,
                    stock_code=None,
                    year=None,
                    report_date=None
                )
                
                if result.get('success'):
                    processed_count += 1
                    # 检查是否是覆盖更新
                    is_update = existing_doc_by_name is not None
                    if is_update:
                        updated_count += 1
                    
                    results.append({
                        'file': pdf_file.name,
                        'status': 'success',
                        'document_id': result.get('document_id'),
                        'chunks_count': result.get('chunks_count', 0),
                        'is_update': is_update
                    })
                    action = "覆盖更新" if is_update else "新增"
                    logger.info(
                        f"文件处理成功 ({action}): {pdf_file.name}, "
                        f"document_id={result.get('document_id')}, "
                        f"chunks={result.get('chunks_count', 0)}"
                    )
                else:
                    failed_count += 1
                    results.append({
                        'file': pdf_file.name,
                        'status': 'failed',
                        'error': result.get('error', '未知错误')
                    })
                    # 如果 ingest_pdf 返回了 document_id，说明可能创建了记录但失败了
                    # 此时 ingest_pdf 内部应该已经删除了记录，但为了保险，这里再检查一次
                    document_id = result.get('document_id')
                    if document_id:
                        try:
                            async with db_manager.get_session() as cleanup_session:
                                stmt = select(Document).where(Document.id == document_id)
                                result_check = await cleanup_session.execute(stmt)
                                doc = result_check.scalar_one_or_none()
                                if doc:
                                    # 如果记录还存在，使用 delete_document_and_related 完整删除
                                    deleted = await self.delete_document_and_related(
                                        cleanup_session, document_id
                                    )
                                    if deleted:
                                        await cleanup_session.commit()
                                        logger.warning(f"清理失败文档记录: document_id={document_id}")
                                    else:
                                        await cleanup_session.rollback()
                                        logger.error(f"清理失败文档记录失败: document_id={document_id}")
                        except Exception as cleanup_error:
                            logger.error(f"清理失败文档记录时出错: {cleanup_error}", exc_info=True)
                    
                    logger.error(f"文件处理失败: {pdf_file.name}, 错误: {result.get('error')}")
            
            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                results.append({
                    'file': pdf_file.name,
                    'status': 'failed',
                    'error': error_msg
                })
                logger.error(f"处理文件时发生异常: {pdf_file.name}, 错误: {error_msg}", exc_info=True)
        
        logger.info(
            f"文件夹扫描完成: {folder_path}, "
            f"处理: {processed_count} (覆盖更新: {updated_count}), 跳过: {skipped_count}, 失败: {failed_count}"
        )
        
        # 所有文件处理完成后，保存索引到文件
        try:
            if self.vector_store.vector_store_type == "faiss":
                self.vector_store.save_index()
                logger.info(f"已保存 FAISS 索引到: {self.vector_store.index_path}")
        except Exception as e:
            logger.error(f"保存索引文件失败: {e}", exc_info=True)
            # 保存失败不影响返回结果
        
        # 记录索引构建到 index_builds 表
        try:
            async with db_manager.get_session() as session:
                # 统计本次构建的文档和 chunks 数量
                total_chunks = sum(r.get('chunks_count', 0) for r in results if r.get('status') == 'success')
                total_docs = processed_count
                
                # 获取配置参数
                retrieval_config = yaml_config.get("retrieval", {})
                params = {
                    "chunk_size": retrieval_config.get("chunk_size", 512),
                    "chunk_overlap": retrieval_config.get("chunk_overlap", 50),
                    "top_k": retrieval_config.get("default_top_k", 5),
                    "rrf_k": retrieval_config.get("rrf_k", 60)
                }
                
                embed_model = yaml_config.get("models", {}).get("default_embedding", "text-embedding-v2")
                
                index_build = IndexBuild(
                    tenant_id=tenant_id,
                    index_name=f"rag_index_{tenant_id}",
                    vector_store=self.vector_store.vector_store_type,
                    embed_model=embed_model,
                    params_json=params,
                    doc_count=total_docs,
                    chunk_count=total_chunks,
                    status="success",
                    finished_at=datetime.now()
                )
                session.add(index_build)
                await session.commit()
                logger.info(f"已记录索引构建: index_build_id={index_build.id}")
        except Exception as e:
            logger.error(f"记录索引构建失败: {e}", exc_info=True)
            # 记录失败不影响返回结果
        
        return {
            'success': True,
            'folder_path': folder_path,
            'total_files': len(pdf_files),
            'processed': processed_count,
            'updated': updated_count,  # 覆盖更新的文件数
            'skipped': skipped_count,
            'failed': failed_count,
            'results': results
        }
