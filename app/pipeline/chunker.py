"""
文本切分器
实现 RecursiveCharacterTextSplitter 逻辑
"""
import re
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """文档切分器，使用 RecursiveCharacterTextSplitter 逻辑"""
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        初始化切分器
        
        Args:
            chunk_size: 块大小（字符数）
            chunk_overlap: 重叠大小（字符数）
            separators: 分隔符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 默认分隔符（按优先级排序）
        # 优化：优先按 \n\n 切分，其次是句号。
        if separators is None:
            separators = [
                "\n\n",  # 段落分隔符（最高优先级）
                "。",    # 中文句号（第二优先级）
                "\n",    # 换行符
                "！",    # 中文感叹号
                "？",    # 中文问号
                ". ",    # 英文句号+空格
                "! ",    # 英文感叹号+空格
                "? ",    # 英文问号+空格
                "；",    # 中文分号
                "; ",    # 英文分号+空格
                "，",    # 中文逗号
                ", ",    # 英文逗号+空格
                " ",     # 空格
                ""       # 字符级别（最后备选）
            ]
        
        self.separators = separators
        
        # 使用 langchain 的 RecursiveCharacterTextSplitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False
        )
    
    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        切分文本
        
        Args:
            text: 要切分的文本
        
        Returns:
            切分后的块列表，每个块包含 {'content': str, 'start': int, 'end': int}
        """
        if not text:
            return []
        
        # 使用 RecursiveCharacterTextSplitter 切分
        chunks = self.splitter.split_text(text)
        
        # 转换为标准格式
        result = []
        current_pos = 0
        
        for chunk in chunks:
            start_pos = text.find(chunk, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(chunk)
            current_pos = end_pos
            
            result.append({
                'content': chunk,
                'start': start_pos,
                'end': end_pos
            })
        
        return result
    
    def split_by_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """
        按段落优先切分，优化规则：优先按 \n\n 切分，其次是句号
        
        Args:
            text: 要切分的文本
        
        Returns:
            切分后的块列表
        """
        if not text:
            return []
        
        # 先按段落分割（\n\n）
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        result = []
        current_pos = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if para_length <= self.chunk_size:
                # 段落可以直接作为一个块
                start_pos = text.find(para, current_pos)
                if start_pos == -1:
                    start_pos = current_pos
                
                result.append({
                    'content': para,
                    'start': start_pos,
                    'end': start_pos + para_length
                })
                current_pos = start_pos + para_length + 2  # +2 for "\n\n"
            else:
                # 段落过长，优先按句号切分
                # 使用正则表达式按句号分割（支持中文和英文句号）
                # 按句号分割，保留句号在句子中
                # 匹配模式：非句号字符 + 句号
                parts = re.split(r'([。.])', para)
                sentences = []
                current_sentence = ""
                
                for i, part in enumerate(parts):
                    if part in ['。', '.']:
                        current_sentence += part
                        if current_sentence.strip():
                            sentences.append(current_sentence.strip())
                        current_sentence = ""
                    else:
                        current_sentence += part
                
                # 添加最后一部分（如果没有以句号结尾）
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                
                # 如果按句号分割后仍然过长，使用递归切分
                if sentences:
                    current_chunk = ""
                    para_start = text.find(para, current_pos)
                    chunk_start = para_start if para_start != -1 else current_pos
                    
                    for sentence in sentences:
                        sentence_len = len(sentence)
                        
                        # 如果当前块加上新句子超过限制
                        if len(current_chunk) + sentence_len + 1 > self.chunk_size and current_chunk:
                            # 保存当前块
                            result.append({
                                'content': current_chunk.strip(),
                                'start': chunk_start,
                                'end': chunk_start + len(current_chunk)
                            })
                            # 开始新块（考虑重叠）
                            overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                            # 计算新块的起始位置（考虑重叠）
                            overlap_start_in_para = para.find(overlap_text, para.find(current_chunk) if current_chunk in para else 0)
                            if overlap_start_in_para != -1 and para_start != -1:
                                chunk_start = para_start + overlap_start_in_para
                            else:
                                # 如果找不到重叠文本，使用段落开始位置
                                chunk_start = para_start if para_start != -1 else current_pos
                            
                            current_chunk = overlap_text + " " + sentence
                        else:
                            # 添加到当前块
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                                # 找到句子在段落中的位置
                                sentence_start_in_para = para.find(sentence)
                                if sentence_start_in_para != -1 and para_start != -1:
                                    chunk_start = para_start + sentence_start_in_para
                                else:
                                    chunk_start = para_start if para_start != -1 else current_pos
                    
                    # 添加最后一个块
                    if current_chunk.strip():
                        result.append({
                            'content': current_chunk.strip(),
                            'start': chunk_start,
                            'end': chunk_start + len(current_chunk)
                        })
                else:
                    # 如果没有句号，使用递归切分
                    sub_chunks = self.split_text(para)
                    for chunk in sub_chunks:
                        chunk['start'] += current_pos
                        chunk['end'] += current_pos
                    result.extend(sub_chunks)
                
                # 更新位置
                para_start = text.find(para, current_pos)
                if para_start != -1:
                    current_pos = para_start + para_length + 2
                else:
                    current_pos += para_length + 2
        
        return result
    
    def split_elements(
        self,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        切分混合元素（文本和表格）
        
        优化规则：
        - 表格：尽量保持完整，除非超长（超过 chunk_size * 2）
        - 文本：优先按 \n\n 切分，其次是句号
        
        Args:
            elements: 元素列表，每个元素包含 {'type': 'text'|'table', 'content': str, ...}
        
        Returns:
            切分后的块列表
        """
        chunks = []
        
        for elem in elements:
            elem_type = elem.get('type', 'text')
            content = elem.get('content', '')
            
            if elem_type == 'table':
                # 表格处理：尽量保持完整
                content_length = len(content)
                max_table_size = self.chunk_size * 2  # 允许表格最大为 chunk_size 的 2 倍
                
                if content_length <= max_table_size:
                    # 表格在允许范围内，保持完整
                    chunks.append({
                        'content': content,
                        'type': 'table',
                        'is_table': True,
                        'table_data': elem.get('table_data'),
                        'page_no': elem.get('page_no'),
                        'metadata': {
                            'element_type': 'table',
                            'bbox': elem.get('bbox')
                        }
                    })
                else:
                    # 表格超长，需要切分（按行切分）
                    # 对于 Markdown 表格，按行分割
                    lines = content.split('\n')
                    current_chunk_lines = []
                    current_chunk_size = 0
                    
                    for line in lines:
                        line_size = len(line) + 1  # +1 for newline
                        
                        if current_chunk_size + line_size > self.chunk_size and current_chunk_lines:
                            # 当前块已满，保存并开始新块
                            chunk_content = '\n'.join(current_chunk_lines)
                            chunks.append({
                                'content': chunk_content,
                                'type': 'table',
                                'is_table': True,
                                'table_data': elem.get('table_data'),  # 保留原始表格数据
                                'page_no': elem.get('page_no'),
                                'metadata': {
                                    'element_type': 'table',
                                    'bbox': elem.get('bbox')
                                }
                            })
                            current_chunk_lines = [line]
                            current_chunk_size = line_size
                        else:
                            current_chunk_lines.append(line)
                            current_chunk_size += line_size
                    
                    # 添加最后一个块
                    if current_chunk_lines:
                        chunk_content = '\n'.join(current_chunk_lines)
                        chunks.append({
                            'content': chunk_content,
                            'type': 'table',
                            'is_table': True,
                            'table_data': elem.get('table_data'),
                            'page_no': elem.get('page_no'),
                            'metadata': {
                                'element_type': 'table',
                                'bbox': elem.get('bbox')
                            }
                        })
            else:
                # 文本元素需要切分
                # 优先按段落切分（\n\n），如果段落过长则按句号切分
                text_chunks = self.split_by_paragraphs(content)
                for chunk in text_chunks:
                    chunks.append({
                        'content': chunk['content'],
                        'type': 'text',
                        'is_table': False,
                        'page_no': elem.get('page_no'),
                        'element_type': elem.get('element_type', 'NarrativeText'),
                        'metadata': {
                            'element_type': elem.get('element_type', 'NarrativeText'),
                            'bbox': elem.get('bbox')
                        }
                    })
        
        return chunks
