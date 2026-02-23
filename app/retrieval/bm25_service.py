"""
BM25 关键词搜索服务
"""
import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank_bm25 not available, BM25 search will be disabled")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("jieba not available, will use simple word splitting")

logger = logging.getLogger(__name__)


class BM25Service:
    """BM25 关键词搜索服务"""
    
    def __init__(self):
        """初始化BM25服务"""
        self.bm25_index = None
        self.chunk_texts = []  # 原始文本列表
        self.chunk_ids = []  # 对应的chunk ID列表
        self.tokenized_corpus = []  # 分词后的语料库
        self._is_initialized = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 输入文本
        
        Returns:
            分词后的词列表
        """
        if JIEBA_AVAILABLE:
            # 使用jieba分词
            words = jieba.cut(text, cut_all=False)
            # 过滤停用词和单字符
            tokens = [w.strip() for w in words if len(w.strip()) > 1]
            return tokens
        else:
            # 简单分词：按空格和标点符号分割
            # 保留中文连续字符和英文单词
            tokens = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+|\d+', text)
            return [t for t in tokens if len(t) > 1]
    
    async def build_index(
        self,
        chunks: List[Dict[str, Any]],
        tenant_id: int = 0
    ):
        """
        构建BM25索引
        
        Args:
            chunks: Chunk数据列表，每个元素包含 {'id': int, 'content': str, ...}
            tenant_id: 租户ID
        """
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, skipping index build")
            return
        
        try:
            self.chunk_texts = []
            self.chunk_ids = []
            self.tokenized_corpus = []
            
            for chunk in chunks:
                content = chunk.get('content', '')
                chunk_id = chunk.get('id')
                
                if not content or not chunk_id:
                    continue
                
                # 分词
                tokens = self._tokenize(content)
                if not tokens:
                    continue
                
                self.chunk_texts.append(content)
                self.chunk_ids.append(chunk_id)
                self.tokenized_corpus.append(tokens)
            
            if self.tokenized_corpus:
                self.bm25_index = BM25Okapi(self.tokenized_corpus)
                self._is_initialized = True
                logger.info(f"Built BM25 index with {len(self.tokenized_corpus)} chunks")
            else:
                logger.warning("No valid chunks for BM25 index")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self._is_initialized = False
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        tenant_id: int = 0
    ) -> List[Dict[str, Any]]:
        """
        BM25关键词搜索
        
        Args:
            query: 查询文本
            top_k: 返回Top-K个结果
            tenant_id: 租户ID
        
        Returns:
            搜索结果列表，每个元素包含 {'id': chunk_id, 'score': bm25_score, ...}
        """
        if not self._is_initialized or self.bm25_index is None:
            logger.warning("BM25 index not initialized, returning empty results")
            return []
        
        try:
            # 对查询进行分词
            query_tokens = self._tokenize(query)
            if not query_tokens:
                logger.warning(f"Query '{query}' has no valid tokens after tokenization")
                return []
            
            # BM25搜索
            scores = self.bm25_index.get_scores(query_tokens)
            
            # 创建结果列表
            results = []
            for i, (chunk_id, score) in enumerate(zip(self.chunk_ids, scores)):
                if score > 0:  # 只返回有分数的结果
                    results.append({
                        'id': chunk_id,
                        'score': float(score),
                        'rank': 0  # 稍后排序
                    })
            
            # 按分数降序排序
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # 更新排名
            for i, result in enumerate(results[:top_k]):
                result['rank'] = i + 1
            
            logger.info(f"BM25 search for '{query[:30]}...' returned {len(results[:top_k])} results")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def is_initialized(self) -> bool:
        """检查索引是否已初始化"""
        return self._is_initialized
