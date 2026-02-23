"""
智能文本切分器
按段落和句子切分，确保切分点不在句子中间
"""
import re
from typing import List, Dict, Any, Optional
from app.config.config import yaml_config


class Splitter:
    """智能文本切分器"""
    
    def __init__(self):
        """初始化切分器"""
        retrieval_config = yaml_config.get("retrieval", {})
        self.chunk_size = retrieval_config.get("chunk_size", 512)
        self.chunk_overlap = retrieval_config.get("chunk_overlap", 50)
        
        # 中文和英文句子结束符
        self.sentence_endings = r'[。！？\n]|[.!?\n]'
        # 段落分隔符
        self.paragraph_separator = r'\n\s*\n'
    
    def split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 使用正则表达式分割句子
        sentences = re.split(self.sentence_endings, text)
        # 过滤空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割成段落"""
        paragraphs = re.split(self.paragraph_separator, text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs
    
    def find_sentence_boundary(
        self,
        text: str,
        start_pos: int,
        max_length: int,
        direction: str = "backward"
    ) -> int:
        """
        在指定位置附近查找句子边界
        
        Args:
            text: 文本
            start_pos: 起始位置
            max_length: 最大长度
            direction: "backward" 或 "forward"
        
        Returns:
            句子边界位置
        """
        if direction == "backward":
            # 向前查找句子结束符
            search_start = max(0, start_pos - max_length)
            search_text = text[search_start:start_pos]
            # 查找最后一个句子结束符
            matches = list(re.finditer(self.sentence_endings, search_text))
            if matches:
                last_match = matches[-1]
                return search_start + last_match.end()
            return search_start
        else:  # forward
            # 向后查找句子结束符
            search_end = min(len(text), start_pos + max_length)
            search_text = text[start_pos:search_end]
            # 查找第一个句子结束符
            match = re.search(self.sentence_endings, search_text)
            if match:
                return start_pos + match.end()
            return search_end
    
    def split_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        智能切分文本
        
        Args:
            text: 要切分的文本
            chunk_size: 块大小（字符数），默认使用配置值
            chunk_overlap: 重叠大小（字符数），默认使用配置值
        
        Returns:
            切分后的块列表，每个块包含 {'content': str, 'start': int, 'end': int}
        """
        if not text:
            return []
        
        chunk_size = chunk_size or self.chunk_size
        chunk_overlap = chunk_overlap or self.chunk_overlap
        
        chunks = []
        current_pos = 0
        text_length = len(text)
        
        while current_pos < text_length:
            # 计算当前块的结束位置
            end_pos = min(current_pos + chunk_size, text_length)
            
            # 如果还没到文本末尾，尝试在句子边界处切分
            if end_pos < text_length:
                # 向后查找句子边界
                boundary_pos = self.find_sentence_boundary(
                    text,
                    end_pos,
                    chunk_size // 4,  # 在1/4块大小范围内查找
                    direction="backward"
                )
                
                # 如果找到边界且不超出范围，使用边界位置
                if boundary_pos > current_pos:
                    end_pos = boundary_pos
                else:
                    # 如果向后找不到，尝试向前查找
                    forward_boundary = self.find_sentence_boundary(
                        text,
                        end_pos,
                        chunk_size // 4,
                        direction="forward"
                    )
                    if forward_boundary <= current_pos + chunk_size * 1.5:
                        end_pos = forward_boundary
            
            # 提取当前块
            chunk_content = text[current_pos:end_pos].strip()
            
            if chunk_content:
                chunks.append({
                    'content': chunk_content,
                    'start': current_pos,
                    'end': end_pos
                })
            
            # 移动到下一个位置（考虑重叠）
            if end_pos >= text_length:
                break
            
            # 计算下一个块的起始位置（考虑重叠）
            next_start = end_pos - chunk_overlap
            # 确保下一个起始位置在句子边界
            if next_start > current_pos:
                boundary_pos = self.find_sentence_boundary(
                    text,
                    next_start,
                    chunk_overlap,
                    direction="backward"
                )
                current_pos = max(current_pos + 1, boundary_pos)
            else:
                current_pos = end_pos
        
        return chunks
    
    def split_by_paragraphs(
        self,
        text: str,
        max_chunk_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        按段落切分，如果段落过长则进一步切分
        
        Args:
            text: 要切分的文本
            max_chunk_size: 最大块大小
        
        Returns:
            切分后的块列表
        """
        max_chunk_size = max_chunk_size or self.chunk_size
        
        paragraphs = self.split_into_paragraphs(text)
        chunks = []
        current_pos = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            if para_length <= max_chunk_size:
                # 段落可以直接作为一个块
                chunks.append({
                    'content': para,
                    'start': current_pos,
                    'end': current_pos + para_length
                })
                current_pos += para_length + 2  # +2 for paragraph separator
            else:
                # 段落过长，需要进一步切分
                sub_chunks = self.split_text(para, max_chunk_size, self.chunk_overlap)
                for chunk in sub_chunks:
                    chunk['start'] += current_pos
                    chunk['end'] += current_pos
                chunks.extend(sub_chunks)
                current_pos += para_length + 2
        
        return chunks
