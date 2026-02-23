"""
PDF 解析器
完全使用 pdfplumber 进行 PDF 解析和表格提取
"""
import hashlib
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pdfplumber


@dataclass
class TableElement:
    """表格元素"""
    page_no: int
    table_data: Dict[str, Any]
    markdown: str
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)


@dataclass
class TextElement:
    """文本元素"""
    page_no: int
    text: str
    element_type: str  # Title, NarrativeText, etc.
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class PageElements:
    """页面元素（文本和表格混合）"""
    page_no: int
    text_elements: List[TextElement]
    table_elements: List[TableElement]
    raw_text: str  # 原始文本（用于备用）


@dataclass
class ParsedPDF:
    """解析后的 PDF 内容"""
    pages: List[PageElements]
    total_pages: int
    file_sha256: str
    file_size: int
    title: Optional[str] = None


class PDFParser:
    """PDF 解析器，完全使用 pdfplumber"""
    
    def __init__(self):
        """初始化解析器"""
        pass
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件的 SHA256 哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def extract_table_with_pdfplumber(self, page) -> List[Dict[str, Any]]:
        """使用 pdfplumber 精准提取表格"""
        tables = []
        
        try:
            page_tables = page.extract_tables()
        except Exception as e:
            # 处理提取表格时的异常（如页面损坏）
            return tables
        
        if not page_tables:
            return tables
        
        for table in page_tables:
            if not table:
                continue
            
            try:
                # 提取表格数据
                rows = []
                for row in table:
                    if row is None:
                        continue
                    cells = [cell.strip() if cell else "" for cell in row]
                    rows.append(cells)
                
                if not rows:
                    continue
                
                # 提取表头（通常第一行）
                header = rows[0] if rows else []
                data = rows[1:] if len(rows) > 1 else []
                
                # 过滤空行
                header = [cell for cell in header if cell]
                data = [[cell for cell in row if cell] for row in data if any(cell for cell in row)]
                
                if not header and not data:
                    continue
                
                table_data = {
                    "header": header,
                    "data": data,
                    "row_count": len(rows),
                    "col_count": len(header) if header else (len(data[0]) if data else 0)
                }
                
                tables.append(table_data)
            except Exception as e:
                # 处理单个表格提取失败的情况
                continue
        
        return tables
    
    def table_to_markdown(self, table_data: Dict[str, Any]) -> str:
        """将表格数据转换为 Markdown 格式"""
        if not table_data:
            return ""
        
        header = table_data.get("header", [])
        data = table_data.get("data", [])
        
        # 如果没有表头，使用第一行数据作为表头
        if not header and data:
            header = data[0] if data else []
            data = data[1:] if len(data) > 1 else []
        
        if not header:
            return ""
        
        lines = []
        # 表头
        lines.append("| " + " | ".join(str(cell) for cell in header) + " |")
        # 分隔线
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        # 数据行
        for row in data:
            row_padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(str(cell) for cell in row_padded[:len(header)]) + " |")
        
        return "\n".join(lines)
    
    def extract_text_from_page(self, page) -> str:
        """从页面提取文本"""
        try:
            text = page.extract_text()
            return text if text else ""
        except Exception as e:
            # 处理提取文本时的异常（如扫描件、损坏页面）
            return ""
    
    def split_text_into_elements(self, text: str, page_no: int) -> List[TextElement]:
        """
        将文本分割为文本元素
        
        简单策略：
        - 按段落分割
        - 识别可能的标题（短行、特定格式）
        """
        if not text or not text.strip():
            return []
        
        elements = []
        
        # 按段落分割
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        for para in paragraphs:
            if not para:
                continue
            
            # 简单判断是否为标题
            # 标题特征：较短、不以句号结尾、可能包含数字编号
            is_title = False
            element_type = "NarrativeText"
            
            lines = para.split("\n")
            first_line = lines[0] if lines else ""
            
            # 标题判断规则
            if len(first_line) < 100 and not first_line.endswith(('。', '.', '；', ';')):
                # 检查是否包含标题关键词或编号
                title_patterns = [
                    r'^第[一二三四五六七八九十\d]+[章节部分]',
                    r'^[一二三四五六七八九十]+、',
                    r'^\d+[\.、]',
                    r'^[（(]\d+[）)]',
                ]
                
                for pattern in title_patterns:
                    if re.match(pattern, first_line):
                        is_title = True
                        element_type = "Title"
                        break
                
                # 如果第一行很短且不是完整句子，可能是标题
                if not is_title and len(first_line) < 50 and len(lines) == 1:
                    is_title = True
                    element_type = "Title"
            
            elements.append(TextElement(
                page_no=page_no,
                text=para,
                element_type=element_type,
                bbox=None  # pdfplumber 不直接提供文本的 bbox
            ))
        
        return elements
    
    def extract_title_from_text(self, text: str) -> Optional[str]:
        """从文本中提取标题（简单实现）"""
        if not text:
            return None
        
        lines = text.strip().split("\n")
        
        # 检查前几行
        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
            
            # 标题通常较短且包含特定关键词
            if len(line) < 200 and any(keyword in line for keyword in ["报告", "公告", "说明", "摘要", "目录"]):
                return line
            
            # 如果第一行非空且较短，可能是标题
            if lines.index(line) < 3 and 10 < len(line) < 100:
                return line
        
        return None
    
    def merge_text_and_tables(
        self,
        text_elements: List[TextElement],
        table_elements: List[TableElement],
        page_no: int
    ) -> List[Dict[str, Any]]:
        """
        合并文本和表格元素，按位置排序
        
        Returns:
            排序后的元素列表，每个元素包含 {'type': 'text'|'table', 'content': str, ...}
        """
        all_elements = []
        
        # 添加文本元素
        for text_elem in text_elements:
            all_elements.append({
                'type': 'text',
                'content': text_elem.text,
                'element_type': text_elem.element_type,
                'page_no': page_no,
                'bbox': text_elem.bbox
            })
        
        # 添加表格元素
        for table_elem in table_elements:
            all_elements.append({
                'type': 'table',
                'content': table_elem.markdown,
                'table_data': table_elem.table_data,
                'page_no': page_no,
                'bbox': table_elem.bbox
            })
        
        # 按位置排序（从上到下，从左到右）
        # 如果 bbox 存在，使用 y 坐标排序
        def sort_key(elem):
            bbox = elem.get('bbox')
            if bbox:
                return (bbox[1], bbox[0])  # (y, x)
            return (0, 0)
        
        all_elements.sort(key=sort_key)
        
        return all_elements
    
    async def parse_pdf(
        self,
        file_path: str,
        extract_tables: bool = True
    ) -> ParsedPDF:
        """
        解析 PDF 文件，完全使用 pdfplumber
        
        Args:
            file_path: PDF 文件路径
            extract_tables: 是否提取表格
        
        Returns:
            ParsedPDF 对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # 计算文件哈希和大小
        file_sha256 = self.calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        
        pages_data = []
        title = None
        
        try:
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
                
                if total_pages == 0:
                    raise ValueError("PDF 文件为空或损坏")
                
                # 按页处理
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        # 提取原始文本
                        raw_text = self.extract_text_from_page(page)
                        
                        # 处理空页或扫描件（无文本）
                        if not raw_text or not raw_text.strip():
                            # 空页或扫描件：创建空的文本元素列表
                            text_elements = []
                            # 记录空页信息（可选）
                            if raw_text == "":
                                # 完全空页
                                pass
                        else:
                            # 分割文本为元素
                            text_elements = self.split_text_into_elements(raw_text, page_num)
                            
                            # 提取标题（从第一页）
                            if page_num == 1 and not title:
                                title = self.extract_title_from_text(raw_text)
                        
                        # 提取表格
                        table_elements = []
                        if extract_tables:
                            try:
                                table_data_list = self.extract_table_with_pdfplumber(page)
                                for table_data in table_data_list:
                                    markdown = self.table_to_markdown(table_data)
                                    if markdown:  # 只添加非空表格
                                        table_elements.append(TableElement(
                                            page_no=page_num,
                                            table_data=table_data,
                                            markdown=markdown
                                        ))
                            except Exception as e:
                                # 表格提取失败，继续处理
                                pass
                        
                        # 如果页面既没有文本也没有表格，至少保留一个空元素
                        if not text_elements and not table_elements:
                            # 空页：创建一个占位文本元素
                            text_elements = [TextElement(
                                page_no=page_num,
                                text="[空页或扫描件，无法提取文本]",
                                element_type="NarrativeText",
                                bbox=None
                            )]
                        
                        pages_data.append(PageElements(
                            page_no=page_num,
                            text_elements=text_elements,
                            table_elements=table_elements,
                            raw_text=raw_text
                        ))
                    
                    except Exception as e:
                        # 处理单个页面解析失败的情况
                        # 创建错误占位元素
                        pages_data.append(PageElements(
                            page_no=page_num,
                            text_elements=[TextElement(
                                page_no=page_num,
                                text=f"[页面解析失败: {str(e)}]",
                                element_type="NarrativeText",
                                bbox=None
                            )],
                            table_elements=[],
                            raw_text=""
                        ))
                        continue
                
                # 如果没有提取到标题，尝试从第一页提取
                if not title and pages_data:
                    first_page = pages_data[0]
                    if first_page.text_elements:
                        # 取第一个文本元素的前200字符作为标题
                        first_text = first_page.text_elements[0].text
                        title = first_text[:200] if len(first_text) > 200 else first_text
        
        except Exception as e:
            raise ValueError(f"PDF 解析失败: {str(e)}")
        
        return ParsedPDF(
            pages=pages_data,
            total_pages=len(pages_data),
            file_sha256=file_sha256,
            file_size=file_size,
            title=title
        )
