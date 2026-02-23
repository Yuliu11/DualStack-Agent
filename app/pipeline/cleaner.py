"""
数据清洗器
去除重复的页眉页脚，保留文档标题结构
"""
import re
from typing import List, Dict, Any
from collections import Counter


class Cleaner:
    """数据清洗器"""
    
    def __init__(self):
        """初始化清洗器"""
        # 常见页眉页脚模式
        self.header_footer_patterns = [
            r'第\s*\d+\s*页\s*共\s*\d+\s*页',  # 页码格式
            r'Page\s+\d+\s+of\s+\d+',  # 英文页码
            r'\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日',  # 日期格式（可能在页脚）
        ]
        
        # 标题模式（用于识别和保留）
        self.title_patterns = [
            r'^第[一二三四五六七八九十\d]+[章节部分]',  # 第X章/节/部分
            r'^[一二三四五六七八九十]+、',  # 一、二、三、
            r'^\d+[\.、]',  # 1. 或 1、
            r'^[（(]\d+[）)]',  # (1) 或 （1）
        ]
    
    def detect_repeating_lines(self, lines: List[str], threshold: int = 3) -> set:
        """
        检测重复出现的行（可能是页眉页脚）
        
        Args:
            lines: 文本行列表
            threshold: 出现次数阈值
        
        Returns:
            重复行的集合
        """
        line_counts = Counter(lines)
        repeating_lines = {
            line for line, count in line_counts.items()
            if count >= threshold and line.strip()  # 非空行且出现多次
        }
        return repeating_lines
    
    def is_header_footer(self, line: str, repeating_lines: set) -> bool:
        """判断是否是页眉页脚"""
        line = line.strip()
        
        # 空行不算
        if not line:
            return False
        
        # 检查是否匹配页眉页脚模式
        for pattern in self.header_footer_patterns:
            if re.search(pattern, line):
                return True
        
        # 检查是否是重复出现的行
        if line in repeating_lines:
            return True
        
        # 检查是否是页码（纯数字或短文本）
        if re.match(r'^\d+$', line) and len(line) < 5:
            return True
        
        # 检查是否是日期格式（可能在页脚）
        if re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$', line):
            return True
        
        return False
    
    def is_title_line(self, line: str) -> bool:
        """判断是否是标题行"""
        line = line.strip()
        
        if not line:
            return False
        
        # 检查是否匹配标题模式
        for pattern in self.title_patterns:
            if re.match(pattern, line):
                return True
        
        # 检查是否是短行且可能是标题（长度在5-100之间，且不以句号结尾）
        if 5 <= len(line) <= 100 and not line.endswith(('。', '.', '；', ';')):
            # 检查是否包含常见标题关键词
            title_keywords = ['报告', '公告', '说明', '摘要', '目录', '概述', '总结']
            if any(keyword in line for keyword in title_keywords):
                return True
        
        return False
    
    def clean_text(self, text: str, preserve_titles: bool = True) -> str:
        """
        清洗文本，去除页眉页脚
        
        Args:
            text: 原始文本
            preserve_titles: 是否保留标题结构
        
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        lines = text.split("\n")
        
        # 检测重复行
        repeating_lines = self.detect_repeating_lines(lines)
        
        cleaned_lines = []
        for line in lines:
            # 跳过页眉页脚
            if self.is_header_footer(line, repeating_lines):
                continue
            
            # 保留标题或普通内容
            if preserve_titles and self.is_title_line(line):
                cleaned_lines.append(line)
            elif not preserve_titles or line.strip():  # 保留非空行
                cleaned_lines.append(line)
        
        # 合并多行，去除多余空行
        cleaned_text = "\n".join(cleaned_lines)
        # 去除连续的空行（最多保留一个）
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
    
    def extract_section_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文档的章节结构
        
        Returns:
            章节结构列表，每个元素包含 {'level': int, 'title': str, 'content': str}
        """
        lines = text.split("\n")
        structure = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 判断标题级别
            level = 0
            if re.match(r'^第[一二三四五六七八九十\d]+[章节部分]', line):
                level = 1
            elif re.match(r'^[一二三四五六七八九十]+、', line):
                level = 2
            elif re.match(r'^\d+[\.、]', line):
                level = 3
            elif re.match(r'^[（(]\d+[）)]', line):
                level = 4
            
            if level > 0:
                # 保存上一章节
                if current_section:
                    structure.append(current_section)
                # 开始新章节
                current_section = {
                    'level': level,
                    'title': line,
                    'content': ""
                }
            elif current_section:
                # 添加到当前章节内容
                current_section['content'] += line + "\n"
        
        # 添加最后一个章节
        if current_section:
            structure.append(current_section)
        
        return structure
