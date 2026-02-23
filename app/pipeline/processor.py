"""
PDF 处理器
支持解析 PDF 文本和表格
"""
import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pdfplumber
from pathlib import Path


@dataclass
class PageContent:
    """页面内容"""
    page_no: int
    text: str
    tables: List[Dict[str, Any]]  # 表格数据


@dataclass
class PDFContent:
    """PDF 内容"""
    pages: List[PageContent]
    total_pages: int
    file_sha256: str
    file_size: int


class PDFProcessor:
    """PDF 处理器，支持文本和表格提取"""
    
    def __init__(self):
        """初始化 PDF 处理器"""
        pass
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件的 SHA256 哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # 分块读取，避免大文件内存问题
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def extract_table_data(self, table) -> Dict[str, Any]:
        """提取表格数据为结构化格式"""
        if not table:
            return {}
        
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() if cell else "" for cell in row]
            rows.append(cells)
        
        # 提取表头（通常第一行）
        header = rows[0] if rows else []
        data = rows[1:] if len(rows) > 1 else []
        
        return {
            "header": header,
            "data": data,
            "row_count": len(rows),
            "col_count": len(header) if header else 0
        }
    
    def table_to_markdown(self, table_data: Dict[str, Any]) -> str:
        """将表格数据转换为 Markdown 格式"""
        if not table_data or not table_data.get("header"):
            return ""
        
        header = table_data["header"]
        data = table_data.get("data", [])
        
        # 构建 Markdown 表格
        lines = []
        # 表头
        lines.append("| " + " | ".join(header) + " |")
        # 分隔线
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        # 数据行
        for row in data:
            # 确保行长度与表头一致
            row_padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(row_padded[:len(header)]) + " |")
        
        return "\n".join(lines)
    
    async def process_pdf(
        self,
        file_path: str,
        extract_tables: bool = True
    ) -> PDFContent:
        """
        处理 PDF 文件，提取文本和表格
        
        Args:
            file_path: PDF 文件路径
            extract_tables: 是否提取表格
        
        Returns:
            PDFContent 对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # 计算文件哈希和大小
        file_sha256 = self.calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        
        pages = []
        
        # 使用 pdfplumber 打开 PDF
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            
            # 逐页处理（支持大文件）
            for page_num, page in enumerate(pdf.pages, start=1):
                # 提取文本
                text = page.extract_text() or ""
                
                # 提取表格
                tables = []
                if extract_tables:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table:
                            table_data = self.extract_table_data(table)
                            if table_data:
                                tables.append(table_data)
                
                pages.append(PageContent(
                    page_no=page_num,
                    text=text,
                    tables=tables
                ))
        
        return PDFContent(
            pages=pages,
            total_pages=total_pages,
            file_sha256=file_sha256,
            file_size=file_size
        )
    
    def extract_title_from_text(self, text: str) -> Optional[str]:
        """从文本中提取标题（简单实现，可根据需要改进）"""
        lines = text.strip().split("\n")
        for line in lines[:10]:  # 检查前10行
            line = line.strip()
            if line and len(line) < 200:  # 标题通常较短
                # 检查是否包含常见标题关键词
                if any(keyword in line for keyword in ["报告", "公告", "说明", "摘要"]):
                    return line
                # 如果第一行非空且较短，可能是标题
                if lines.index(line) < 3 and 10 < len(line) < 100:
                    return line
        return None
