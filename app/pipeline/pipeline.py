"""
文档摄取管道主流程
整合 PDF 处理、清洗、切分、增量更新和持久化
"""
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
from pathlib import Path

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.pipeline.processor import PDFProcessor, PDFContent
from app.pipeline.cleaner import Cleaner
from app.pipeline.splitter import Splitter
from app.storage.db_manager import db_manager
from app.storage.models import Document, DocVersion, Chunk, Table
from app.config.config import yaml_config


class DocumentPipeline:
    """文档摄取管道"""
    
    def __init__(self):
        """初始化管道"""
        self.processor = PDFProcessor()
        self.cleaner = Cleaner()
        self.splitter = Splitter()
        self.default_tenant_id = 0
    
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
    
    async def find_or_create_document(
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
    ) -> Tuple[Document, bool]:
        """
        查找或创建文档记录
        
        Returns:
            (Document, is_new) 元组
        """
        # 查找是否已存在
        stmt = select(Document).where(
            and_(
                Document.tenant_id == tenant_id,
                Document.file_sha256 == file_sha256
            )
        ).limit(1)
        
        result = await session.execute(stmt)
        existing_doc = result.scalar_one_or_none()
        
        if existing_doc:
            # 更新解析状态
            existing_doc.parse_status = "parsed"
            existing_doc.parse_error = None
            return existing_doc, False
        
        # 创建新文档
        new_doc = Document(
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
            parse_status="parsed"
        )
        session.add(new_doc)
        await session.flush()  # 获取 ID
        
        return new_doc, True
    
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
            markdown_content = self.processor.table_to_markdown(table_data)
            return existing_table, markdown_content
        
        # 创建新表格
        table = Table(
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
            page_no=page_no,
            table_title=None,  # 可以从表格数据中提取
            schema_json=table_data.get("header"),  # 表头作为 schema
            data_json=table_data,
            table_sha256=table_sha256
        )
        session.add(table)
        await session.flush()  # 获取 ID
        
        # 转换为 Markdown
        markdown_content = self.processor.table_to_markdown(table_data)
        
        return table, markdown_content
    
    async def process_chunks(
        self,
        session: AsyncSession,
        chunks_data: List[Dict[str, Any]],
        document_id: int,
        version_id: int,
        tenant_id: int = 0
    ) -> Tuple[List[Chunk], int]:
        """
        处理 chunks，实现增量更新
        
        Returns:
            (创建的 Chunk 对象列表, 复用的 chunk 数量)
        """
        created_chunks = []
        reused_count = 0
        
        for idx, chunk_data in enumerate(chunks_data, start=1):
            content = chunk_data['content']
            content_sha256 = self.calculate_content_hash(content)
            
            # 检查是否已存在
            existing_chunk = await self.find_existing_chunk(
                session,
                tenant_id,
                content_sha256
            )
            
            if existing_chunk:
                # 复用已存在的 chunk：创建新记录但内容相同
                # 这样可以保持版本独立性，同时避免重复计算嵌入向量
                # 注意：这里我们仍然创建新记录，因为不同版本可能有不同的元数据
                # 但可以通过 content_sha256 关联，后续嵌入计算时可以复用
                reused_count += 1
            
            # 创建新 chunk（即使是复用，也创建新记录以保持版本独立性）
            chunk = Chunk(
                tenant_id=tenant_id,
                document_id=document_id,
                version_id=version_id,
                chunk_no=idx,
                page_start=chunk_data.get('page_start'),
                page_end=chunk_data.get('page_end'),
                section_path=chunk_data.get('section_path'),
                content=content,
                content_sha256=content_sha256,
                content_tokens=None,  # 可以后续计算
                is_table=chunk_data.get('is_table', False),
                table_id=chunk_data.get('table_id'),
                is_deleted=False
            )
            session.add(chunk)
            created_chunks.append(chunk)
        
        await session.flush()  # 获取所有 ID
        return created_chunks, reused_count
    
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
        摄取 PDF 文档的主流程
        
        Args:
            file_path: PDF 文件路径
            tenant_id: 租户ID
            doc_type: 文档类型
            source: 来源
            title: 文档标题（如果为None，会尝试从PDF提取）
            company_name: 公司名称
            stock_code: 股票代码
            year: 年份
            report_date: 报告日期
            version_tag: 版本标签
        
        Returns:
            处理结果字典
        """
        try:
            # 1. 处理 PDF
            pdf_content = await self.processor.process_pdf(file_path, extract_tables=True)
            
            # 2. 提取标题（如果未提供）
            if not title:
                # 尝试从前几页提取标题
                if pdf_content.pages:
                    first_page_text = pdf_content.pages[0].text
                    title = self.processor.extract_title_from_text(first_page_text) or Path(file_path).stem
            
            # 3. 清洗和切分
            all_chunks_data = []
            all_tables = []
            
            for page_content in pdf_content.pages:
                # 清洗文本
                cleaned_text = self.cleaner.clean_text(page_content.text, preserve_titles=True)
                
                # 切分文本
                text_chunks = self.splitter.split_text(cleaned_text)
                
                # 转换为 chunks 数据格式
                for chunk in text_chunks:
                    all_chunks_data.append({
                        'content': chunk['content'],
                        'page_start': page_content.page_no,
                        'page_end': page_content.page_no,
                        'section_path': None,  # 可以从标题结构提取
                        'is_table': False
                    })
                
                # 处理表格
                for table_data in page_content.tables:
                    all_tables.append({
                        'data': table_data,
                        'page_no': page_content.page_no
                    })
            
            # 4. 持久化到数据库
            async with db_manager.get_session() as session:
                # 创建或查找文档
                document, is_new = await self.find_or_create_document(
                    session,
                    file_uri=file_path,
                    file_sha256=pdf_content.file_sha256,
                    file_size=pdf_content.file_size,
                    title=title,
                    tenant_id=tenant_id,
                    doc_type=doc_type,
                    source=source,
                    company_name=company_name,
                    stock_code=stock_code,
                    year=year,
                    report_date=report_date
                )
                
                # 确定版本号
                if is_new:
                    version_no = 1
                else:
                    # 查找最大版本号
                    stmt = select(DocVersion.version_no).where(
                        DocVersion.document_id == document.id
                    ).order_by(DocVersion.version_no.desc()).limit(1)
                    result = await session.execute(stmt)
                    max_version = result.scalar_one_or_none()
                    version_no = (max_version or 0) + 1
                
                # 计算整体内容哈希（所有chunks的哈希）
                all_content = "\n".join([c['content'] for c in all_chunks_data])
                content_sha256 = self.calculate_content_hash(all_content)
                
                # 创建文档版本
                version = await self.create_document_version(
                    session,
                    document_id=document.id,
                    version_no=version_no,
                    content_sha256=content_sha256,
                    tenant_id=tenant_id,
                    version_tag=version_tag
                )
                
                # 处理表格
                table_id_map = {}  # 用于关联 chunk 和 table
                for table_info in all_tables:
                    table_obj, markdown_content = await self.process_table(
                        session,
                        table_info['data'],
                        document_id=document.id,
                        version_id=version.id,
                        page_no=table_info['page_no'],
                        tenant_id=tenant_id
                    )
                    table_id_map[table_obj.id] = markdown_content
                
                # 处理表格 chunks（将表格作为特殊 chunk）
                for table_id, markdown_content in table_id_map.items():
                    table_chunk_data = {
                        'content': markdown_content,
                        'page_start': None,
                        'page_end': None,
                        'section_path': None,
                        'is_table': True,
                        'table_id': table_id
                    }
                    all_chunks_data.append(table_chunk_data)
                
                # 处理文本 chunks
                chunks, reused_chunks_count = await self.process_chunks(
                    session,
                    all_chunks_data,
                    document_id=document.id,
                    version_id=version.id,
                    tenant_id=tenant_id
                )
                
                # 更新文档的当前版本ID
                document.current_version_id = version.id
                
                await session.commit()
                
                return {
                    'success': True,
                    'document_id': document.id,
                    'version_id': version.id,
                    'version_no': version_no,
                    'chunks_count': len(chunks),
                    'reused_chunks_count': reused_chunks_count,
                    'tables_count': len(all_tables),
                    'file_sha256': pdf_content.file_sha256
                }
        
        except Exception as e:
            # 记录错误
            error_msg = str(e)
            
            # 尝试更新文档状态为失败
            try:
                # 如果 PDF 处理成功，尝试计算文件哈希
                file_sha256 = None
                if 'pdf_content' in locals():
                    file_sha256 = pdf_content.file_sha256
                else:
                    # 如果 PDF 处理失败，尝试直接计算文件哈希
                    try:
                        file_sha256 = self.processor.calculate_file_hash(file_path)
                    except:
                        pass
                
                if file_sha256:
                    async with db_manager.get_session() as session:
                        # 查找文档（如果已创建）
                        stmt = select(Document).where(
                            and_(
                                Document.tenant_id == tenant_id,
                                Document.file_sha256 == file_sha256
                            )
                        ).limit(1)
                        result = await session.execute(stmt)
                        doc = result.scalar_one_or_none()
                        if doc:
                            doc.parse_status = "failed"
                            doc.parse_error = error_msg[:65535]  # TEXT 字段限制
                            await session.commit()
            except Exception as update_error:
                # 更新文档状态失败不影响错误返回
                pass
            
            return {
                'success': False,
                'error': error_msg
            }
