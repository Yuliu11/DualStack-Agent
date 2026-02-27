"""
文档摄取管道模块
"""
from app.pipeline.ingestion import DocumentIngestionPipeline
from app.pipeline.parser import PDFParser
from app.pipeline.cleaner import Cleaner
from app.pipeline.chunker import DocumentChunker
from app.pipeline.query import QueryRewriter, query_rewriter, SlotExtractor, slot_extractor

# 创建全局单例
document_pipeline = DocumentIngestionPipeline()

__all__ = [
    "DocumentIngestionPipeline",
    "document_pipeline",
    "PDFParser",
    "Cleaner",
    "DocumentChunker",
    "QueryRewriter",
    "query_rewriter",
    "SlotExtractor",
    "slot_extractor",
]