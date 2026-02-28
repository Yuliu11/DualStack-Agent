"""
PDF 解析器
完全使用 pdfplumber 进行 PDF 解析和表格提取。
支持跨页表格：维护表头状态，对页面顶部的续表注入虚拟表头。
"""
import hashlib
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import pdfplumber

logger = logging.getLogger(__name__)

# 表头常见关键词（用于判定第一行是否为表头）
HEADER_KEYWORDS = frozenset({
    "项目", "年度", "金额", "科目", "序号", "名称", "单位", "日期", "合计",
    "备注", "说明", "单位：", "万元", "元", "比例", "占比", "增长率",
    "期末", "期初", "本期", "上期", "同比", "环比", "代码", "数量",
    "金额（", "比例（", "名称（", "项目（", "行次", "列", "行",
})
# 页面顶部阈值（pt）：表格上边界低于此值视为“页面顶部”，可能是跨页续表
TABLE_TOP_Y_THRESHOLD = 180


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
    """PDF 解析器，完全使用 pdfplumber，支持跨页表头维护与注入"""
    
    def __init__(self):
        """初始化解析器"""
        pass
    
    @staticmethod
    def _is_header_like(first_row: List[str]) -> bool:
        """
        判断第一行是否像表头（包含“项目”、“年度”、“金额”等）。
        表头特征：存在单元格包含常见表头关键词，或整行多为短文本。
        """
        if not first_row:
            return False
        cells = [str(c).strip() for c in first_row if c]
        if not cells:
            return False
        # 任一单元格包含表头关键词
        for cell in cells:
            for kw in HEADER_KEYWORDS:
                if kw in cell:
                    return True
        # 表头多为较短文本；若整行单元格都较短且不全为数字，也倾向视为表头
        all_short = all(len(str(c)) <= 20 for c in cells)
        not_all_numeric = not all(
            re.match(r"^[\d.,%\s\-]+$", str(c).strip()) for c in cells
        )
        if all_short and not_all_numeric and len(cells) >= 2:
            return True
        return False
    
    @staticmethod
    def _is_data_like(first_row: List[str]) -> bool:
        """判断第一行是否像数据行（非表头）：多为数字或较长文本。"""
        if not first_row:
            return True
        cells = [str(c).strip() for c in first_row if c]
        if not cells:
            return True
        # 若已判定为表头则不是数据
        if PDFParser._is_header_like(first_row):
            return False
        # 多数为数字或较长字符串则视为数据
        numeric_or_long = sum(
            1 for c in cells
            if re.match(r"^[\d.,%\s\-]+$", str(c)) or len(str(c)) > 20
        )
        return numeric_or_long >= len(cells) / 2
    
    @staticmethod
    def _is_table_at_page_top(bbox: Optional[Tuple[float, ...]], page_height: float) -> bool:
        """判断表格是否位于页面顶部（y 较小）。用于识别跨页续表。"""
        if not bbox or len(bbox) < 4:
            return False
        top_y = bbox[1]  # pdfplumber bbox: (x0, top, x1, bottom)
        return top_y < TABLE_TOP_Y_THRESHOLD
    
    @staticmethod
    def _clean_page_text(raw_text: str) -> str:
        """
        移除页眉、页脚、页码等重复噪音，避免夹在跨页表格中间干扰语义。
        - 移除单独成行的纯数字（页码）
        - 移除常见页眉页脚短行（如 "第 X 页"）
        """
        if not raw_text or not raw_text.strip():
            return raw_text
        lines = raw_text.split("\n")
        cleaned = []
        for line in lines:
            s = line.strip()
            # 跳过纯数字行（页码）
            if re.match(r"^\d{1,5}$", s):
                continue
            # 跳过 "第 N 页" / "Page N"
            if re.match(r"^(第\s*\d+\s*页|Page\s*\d+)$", s, re.IGNORECASE):
                continue
            # 跳过仅含数字和空格/标点的短行（可能是页码变体）
            if len(s) <= 10 and re.match(r"^[\d\s./\-]+$", s):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)
    
    def calculate_file_hash(self, file_path: str) -> str:
        """计算文件的 SHA256 哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_tables_with_bbox(
        self, page
    ) -> List[Tuple[Dict[str, Any], Optional[Tuple[float, float, float, float]]]]:
        """
        提取当前页所有表格，并返回 (table_data, bbox) 列表。
        优先使用 find_tables() 以获取 bbox（用于判断是否在页面顶部），
        若无 bbox 则回退到 extract_tables()。
        """
        result: List[Tuple[Dict[str, Any], Optional[Tuple[float, float, float, float]]]] = []
        try:
            found = page.find_tables()
            if found:
                for tbl in found:
                    try:
                        raw = tbl.extract()
                        if not raw:
                            continue
                        rows = []
                        for row in raw:
                            if row is None:
                                continue
                            cells = [cell.strip() if cell else "" for cell in row]
                            rows.append(cells)
                        if not rows:
                            continue
                        header = rows[0]
                        data = rows[1:] if len(rows) > 1 else []
                        header = [c for c in header if c]
                        data = [[c for c in row if c] for row in data if any(c for c in row)]
                        if not header and not data:
                            continue
                        table_data = {
                            "header": header,
                            "data": data,
                            "row_count": len(rows),
                            "col_count": len(header) if header else (len(data[0]) if data else 0),
                        }
                        bbox = tbl.bbox if hasattr(tbl, "bbox") else None
                        result.append((table_data, bbox))
                    except Exception as e:
                        logger.debug("单表提取失败: %s", e)
                        continue
            if not result:
                # 回退：无 bbox
                page_tables = page.extract_tables()
                for table in page_tables or []:
                    if not table:
                        continue
                    try:
                        rows = []
                        for row in table:
                            if row is None:
                                continue
                            cells = [cell.strip() if cell else "" for cell in row]
                            rows.append(cells)
                        if not rows:
                            continue
                        header = rows[0]
                        data = rows[1:] if len(rows) > 1 else []
                        header = [c for c in header if c]
                        data = [[c for c in row if c] for row in data if any(c for c in row)]
                        if not header and not data:
                            continue
                        table_data = {
                            "header": header,
                            "data": data,
                            "row_count": len(rows),
                            "col_count": len(header) if header else (len(data[0]) if data else 0),
                        }
                        result.append((table_data, None))
                    except Exception as e:
                        logger.debug("单表提取失败(回退): %s", e)
        except Exception as e:
            logger.warning("本页表格提取异常: %s", e)
        return result
    
    def _apply_cross_page_header(
        self,
        table_data: Dict[str, Any],
        bbox: Optional[Tuple[float, ...]],
        page_height: float,
        page_no: int,
        last_table_header: Optional[List[str]],
    ) -> Tuple[Dict[str, Any], Optional[List[str]]]:
        """
        根据表头状态处理当前表格：新表则更新表头；跨页续表则注入 last_table_header。
        返回 (修复后的 table_data, 更新后的 last_table_header)。
        """
        header = table_data.get("header", [])
        data = table_data.get("data", [])
        first_row = header if header else (data[0] if data else [])
        if not first_row:
            return table_data, last_table_header
        
        # 情况 A：第一行像表头 -> 视为新表格，更新 last_table_header
        if self._is_header_like(first_row):
            logger.info(
                "[表头] 页 %d 识别为新表头，列数=%d，首列示例=%s",
                page_no,
                len(first_row),
                (first_row[0][:30] + "…") if len(first_row[0]) > 30 else first_row[0],
            )
            new_header = [str(c).strip() for c in first_row]
            return table_data, new_header
        
        # 情况 B：第一行像数据 且 表格在页面顶部 且 有可用的上次表头 -> 跨页延续，注入表头
        at_top = self._is_table_at_page_top(bbox, page_height)
        if (
            self._is_data_like(first_row)
            and at_top
            and last_table_header is not None
            and len(last_table_header) > 0
        ):
            # 将当前第一行归入 data，表头用 last_table_header；若列数不一致则按 last_table_header 列数截断/补齐
            injected_header = list(last_table_header)
            new_data = [list(first_row)] + data
            col_count = len(injected_header)
            new_data = [
                (row + [""] * (col_count - len(row)))[:col_count]
                for row in new_data
            ]
            fixed = {
                "header": injected_header,
                "data": new_data,
                "row_count": len(new_data) + 1,
                "col_count": col_count,
            }
            logger.info(
                "[表头] 页 %d 跨页续表：注入 %d 列表头，原首行作为数据行",
                page_no,
                len(injected_header),
            )
            return fixed, last_table_header
        
        # 其他：页面顶部无表格或非续表，若有 last_table_header 且当前无有效表头也可注入（保证 chunk 带表头）
        if last_table_header and (not header or not self._is_header_like(header)):
            injected_header = list(last_table_header)
            all_rows = (header if header else []) + data
            if all_rows:
                col_count = len(injected_header)
                new_data = [
                    (list(row) + [""] * (col_count - len(row)))[:col_count]
                    for row in all_rows
                ]
                fixed = {
                    "header": injected_header,
                    "data": new_data,
                    "row_count": len(new_data) + 1,
                    "col_count": col_count,
                }
                logger.debug("[表头] 页 %d 无表头，使用上次表头补齐", page_no)
                return fixed, last_table_header
        
        return table_data, last_table_header
    
    def extract_table_with_pdfplumber(self, page) -> List[Dict[str, Any]]:
        """使用 pdfplumber 精准提取表格（兼容旧接口：仅返回 table_data 列表，无跨页逻辑）"""
        pairs = self._extract_tables_with_bbox(page)
        return [t[0] for t in pairs]
    
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
        
        if extract_tables:
            logger.info("PDF 解析启用表格提取与跨页表头注入（last_table_header 维护）")
        
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
                
                # 跨页表头状态：遍历页面时维护最近一次识别到的完整表头
                last_table_header: Optional[List[str]] = None
                
                # 按页处理
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        # 提取原始文本并清理页眉页脚、页码
                        raw_text = self.extract_text_from_page(page)
                        cleaned_text = self._clean_page_text(raw_text)
                        
                        # 处理空页或扫描件（无文本）
                        if not cleaned_text or not cleaned_text.strip():
                            text_elements = []
                            if raw_text == "":
                                pass
                        else:
                            text_elements = self.split_text_into_elements(cleaned_text, page_num)
                            if page_num == 1 and not title:
                                title = self.extract_title_from_text(cleaned_text)
                        
                        # 表格提取：带 bbox，并应用跨页表头注入
                        table_elements = []
                        if extract_tables:
                            try:
                                page_height = float(getattr(page, "height", 842) or 842)
                                table_pairs = self._extract_tables_with_bbox(page)
                                for table_data, bbox in table_pairs:
                                    fixed_data, last_table_header = self._apply_cross_page_header(
                                        table_data,
                                        bbox,
                                        page_height,
                                        page_num,
                                        last_table_header,
                                    )
                                    markdown = self.table_to_markdown(fixed_data)
                                    if markdown:
                                        table_elements.append(TableElement(
                                            page_no=page_num,
                                            table_data=fixed_data,
                                            markdown=markdown,
                                            bbox=bbox,
                                        ))
                            except Exception as e:
                                logger.warning("页 %d 表格提取失败: %s", page_num, e)
                        
                        if not text_elements and not table_elements:
                            text_elements = [TextElement(
                                page_no=page_num,
                                text="[空页或扫描件，无法提取文本]",
                                element_type="NarrativeText",
                                bbox=None
                            )]
                        
                        # 存储使用清理后的文本，避免页眉页脚进入后续 chunk
                        pages_data.append(PageElements(
                            page_no=page_num,
                            text_elements=text_elements,
                            table_elements=table_elements,
                            raw_text=cleaned_text
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
