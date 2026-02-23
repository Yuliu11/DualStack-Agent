"""
数据库 ORM 模型
使用 SQLAlchemy 2.0 异步语法（Mapped/mapped_column）
"""
from datetime import datetime, date
from typing import Optional
from decimal import Decimal
from sqlalchemy import (
    BigInteger, String, Text, Integer, Date, DateTime, 
    Float, Boolean, JSON, ForeignKey, UniqueConstraint, Index,
    ForeignKeyConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.mysql import TINYINT, LONGTEXT, MEDIUMTEXT, DECIMAL, BIGINT
from app.storage.db_manager import Base


class Document(Base):
    """文档表"""
    __tablename__ = "documents"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    tenant_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        default=0,
        index=True
    )
    doc_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="annual_report"
    )
    source: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="cninfo"
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    company_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    stock_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    report_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    file_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column(BIGINT(unsigned=True), nullable=True)
    
    parse_status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending"
    )
    parse_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    current_version_id: Mapped[Optional[int]] = mapped_column(
        BIGINT(unsigned=True),
        nullable=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"
    )
    
    # 关系
    versions: Mapped[list["DocVersion"]] = relationship(
        "DocVersion",
        back_populates="document"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document"
    )
    tables: Mapped[list["Table"]] = relationship(
        "Table",
        back_populates="document"
    )
    
    __table_args__ = (
        UniqueConstraint("tenant_id", "file_sha256", name="uk_tenant_filehash"),
        Index("idx_company_year", "tenant_id", "stock_code", "year"),
    )


class DocVersion(Base):
    """文档版本表"""
    __tablename__ = "doc_versions"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    tenant_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        default=0
    )
    document_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        index=True
    )
    
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)
    version_tag: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP"
    )
    
    # 关系
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="versions",
        foreign_keys=[document_id]
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="version"
    )
    tables: Mapped[list["Table"]] = relationship(
        "Table",
        back_populates="version"
    )
    
    __table_args__ = (
        UniqueConstraint("document_id", "version_no", name="uk_doc_ver"),
        ForeignKeyConstraint(["document_id"], ["documents.id"], name="fk_ver_doc"),
    )


class Chunk(Base):
    """文档块表"""
    __tablename__ = "chunks"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    tenant_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        default=0
    )
    document_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("documents.id"),
        nullable=False,
        index=True
    )
    version_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("doc_versions.id"),
        nullable=False,
        index=True
    )
    
    chunk_no: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    page_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    content: Mapped[str] = mapped_column(LONGTEXT, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    
    content_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_table: Mapped[bool] = mapped_column(
        TINYINT(1),
        nullable=False,
        default=False
    )
    table_id: Mapped[Optional[int]] = mapped_column(
        BIGINT(unsigned=True),
        nullable=True
    )
    
    is_deleted: Mapped[bool] = mapped_column(
        TINYINT(1),
        nullable=False,
        default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP"
    )
    
    # 关系
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
        foreign_keys=[document_id]
    )
    version: Mapped["DocVersion"] = relationship(
        "DocVersion",
        back_populates="chunks",
        foreign_keys=[version_id]
    )
    embeddings: Mapped[list["Embedding"]] = relationship(
        "Embedding",
        back_populates="chunk",
        foreign_keys="Embedding.chunk_id"
    )
    citations: Mapped[list["QACitation"]] = relationship(
        "QACitation",
        back_populates="chunk",
        foreign_keys="QACitation.chunk_id"
    )
    
    __table_args__ = (
        UniqueConstraint("version_id", "chunk_no", name="uk_ver_chunkno"),
        Index("idx_doc_ver", "document_id", "version_id"),
        Index("idx_chunk_hash", "tenant_id", "content_sha256"),
        Index("ft_content", "content", mysql_prefix="FULLTEXT"),
        ForeignKeyConstraint(["document_id"], ["documents.id"], name="fk_chunk_doc"),
        ForeignKeyConstraint(["version_id"], ["doc_versions.id"], name="fk_chunk_ver"),
    )


class Table(Base):
    """表格表"""
    __tablename__ = "tables"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    tenant_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        default=0
    )
    document_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("documents.id"),
        nullable=False,
        index=True
    )
    version_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("doc_versions.id"),
        nullable=False,
        index=True
    )
    
    page_no: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    table_title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    schema_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    data_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    table_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP"
    )
    
    # 关系
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="tables",
        foreign_keys=[document_id]
    )
    version: Mapped["DocVersion"] = relationship(
        "DocVersion",
        back_populates="tables",
        foreign_keys=[version_id]
    )
    
    __table_args__ = (
        Index("idx_doc_ver", "document_id", "version_id"),
        Index("idx_table_hash", "tenant_id", "table_sha256"),
        ForeignKeyConstraint(["document_id"], ["documents.id"], name="fk_table_doc"),
        ForeignKeyConstraint(["version_id"], ["doc_versions.id"], name="fk_table_ver"),
    )


class Embedding(Base):
    """向量嵌入表"""
    __tablename__ = "embeddings"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    tenant_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        default=0
    )
    chunk_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("chunks.id"),
        nullable=False,
        index=True
    )
    
    embed_model: Mapped[str] = mapped_column(String(128), nullable=False)
    embed_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_store: Mapped[str] = mapped_column(String(32), nullable=False)
    vector_id: Mapped[str] = mapped_column(String(128), nullable=False)
    
    embed_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP"
    )
    
    # 关系
    chunk: Mapped["Chunk"] = relationship(
        "Chunk",
        back_populates="embeddings",
        foreign_keys=[chunk_id]
    )
    
    __table_args__ = (
        UniqueConstraint("chunk_id", "embed_model", name="uk_chunk_model"),
        Index("idx_vector", "vector_store", "vector_id"),
        Index("idx_embed_hash", "tenant_id", "embed_sha256"),
        ForeignKeyConstraint(["chunk_id"], ["chunks.id"], name="fk_embed_chunk"),
    )


class IndexBuild(Base):
    """索引构建表"""
    __tablename__ = "index_builds"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    tenant_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        default=0
    )
    
    index_name: Mapped[str] = mapped_column(String(128), nullable=False)
    vector_store: Mapped[str] = mapped_column(String(32), nullable=False)
    embed_model: Mapped[str] = mapped_column(String(128), nullable=False)
    
    params_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    doc_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="running"
    )
    error_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP"
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    __table_args__ = (
        Index("idx_index_name", "tenant_id", "index_name"),
        Index("idx_status", "status"),
    )


class QARequest(Base):
    """问答请求表"""
    __tablename__ = "qa_requests"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    tenant_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        default=0
    )
    user_id: Mapped[Optional[int]] = mapped_column(
        BIGINT(unsigned=True),
        nullable=True
    )
    
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expanded_query_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    retrieval_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    answer_text: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)
    answer_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    answer_policy: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="cite_required"
    )
    
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_refused: Mapped[bool] = mapped_column(
        TINYINT(1),
        nullable=False,
        default=False
    )
    
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[Optional[Decimal]] = mapped_column(
        DECIMAL(10, 6),
        nullable=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP",
        index=True
    )
    
    # 关系
    citations: Mapped[list["QACitation"]] = relationship(
        "QACitation",
        back_populates="request",
        foreign_keys="QACitation.request_id"
    )
    model_events: Mapped[list["ModelEvent"]] = relationship(
        "ModelEvent",
        back_populates="request",
        foreign_keys="ModelEvent.request_id"
    )
    
    __table_args__ = (
        Index("idx_queryhash", "tenant_id", "query_hash"),
    )


class QACitation(Base):
    """问答引用表"""
    __tablename__ = "qa_citations"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    request_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("qa_requests.id"),
        nullable=False,
        index=True
    )
    chunk_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("chunks.id"),
        nullable=False,
        index=True
    )
    rank_no: Mapped[int] = mapped_column(Integer, nullable=False)
    quote_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP"
    )
    
    # 关系
    request: Mapped["QARequest"] = relationship(
        "QARequest",
        back_populates="citations",
        foreign_keys=[request_id]
    )
    chunk: Mapped["Chunk"] = relationship(
        "Chunk",
        back_populates="citations",
        foreign_keys=[chunk_id]
    )
    
    __table_args__ = (
        Index("idx_req", "request_id"),
        ForeignKeyConstraint(["request_id"], ["qa_requests.id"], name="fk_cite_req"),
        ForeignKeyConstraint(["chunk_id"], ["chunks.id"], name="fk_cite_chunk"),
    )


class ModelEvent(Base):
    """模型事件表"""
    __tablename__ = "model_events"
    
    id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        primary_key=True,
        autoincrement=True
    )
    tenant_id: Mapped[int] = mapped_column(
        BIGINT(unsigned=True),
        nullable=False,
        default=0
    )
    request_id: Mapped[Optional[int]] = mapped_column(
        BIGINT(unsigned=True),
        ForeignKey("qa_requests.id"),
        nullable=True,
        index=True
    )
    
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    http_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    error_msg: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fallback_to: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default="CURRENT_TIMESTAMP",
        index=True
    )
    
    # 关系
    request: Mapped[Optional["QARequest"]] = relationship(
        "QARequest",
        back_populates="model_events",
        foreign_keys=[request_id]
    )
    
    __table_args__ = (
        Index("idx_req", "request_id"),
        Index("idx_model_time", "model_name", "created_at"),
        ForeignKeyConstraint(["request_id"], ["qa_requests.id"], name="fk_evt_req"),
    )
