"""
向量存储管理
支持 FAISS 和 Qdrant
"""
import os
import pickle
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, vector search will be disabled")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logger.warning("Qdrant not available, will use FAISS only")


class VectorStore:
    """向量存储管理器"""
    
    def __init__(self, vector_store_type: str = "faiss", index_path: Optional[str] = None):
        """
        初始化向量存储
        
        Args:
            vector_store_type: 向量存储类型 ("faiss" 或 "qdrant")
            index_path: FAISS索引文件路径（仅用于FAISS）
        """
        self.vector_store_type = vector_store_type
        self.index_path = index_path or "./data/faiss_index"
        self.index = None
        self.id_to_chunk = {}  # vector_id -> chunk_id 映射
        self.chunk_to_id = {}  # chunk_id -> vector_id 映射
        
        if vector_store_type == "faiss" and FAISS_AVAILABLE:
            self._load_faiss_index()
        elif vector_store_type == "qdrant" and QDRANT_AVAILABLE:
            self._init_qdrant()
        else:
            logger.warning(f"Vector store type {vector_store_type} not available, using in-memory storage")
            self.vector_store_type = "memory"
            self.vectors = {}  # vector_id -> vector
            self.vector_dim = None
    
    def _load_faiss_index(self):
        """加载FAISS索引"""
        index_file = Path(self.index_path)
        if index_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                # 加载ID映射
                mapping_file = index_file.parent / f"{index_file.name}.mapping"
                if mapping_file.exists():
                    with open(mapping_file, 'rb') as f:
                        self.id_to_chunk = pickle.load(f)
                        self.chunk_to_id = {v: k for k, v in self.id_to_chunk.items()}
                logger.info(f"Loaded FAISS index from {index_file}, {len(self.id_to_chunk)} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.index = None
        else:
            logger.info(f"FAISS index not found at {index_file}, will create new index when needed")
    
    def _init_qdrant(self):
        """初始化Qdrant客户端"""
        from app.config.config import yaml_config
        qdrant_config = yaml_config.get("vector_store", {}).get("qdrant", {})
        url = qdrant_config.get("url", "http://localhost:6333")
        collection_name = qdrant_config.get("collection_name", "rag_collection")
        
        try:
            self.qdrant_client = QdrantClient(url=url)
            self.collection_name = collection_name
            logger.info(f"Initialized Qdrant client: {url}, collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            self.qdrant_client = None
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        tenant_id: int = 0
    ) -> List[Dict[str, Any]]:
        """
        向量搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回Top-K个结果
            tenant_id: 租户ID
        
        Returns:
            搜索结果列表，每个元素包含 {'id': chunk_id, 'score': similarity_score, ...}
        """
        if self.vector_store_type == "faiss" and self.index is not None:
            return self._search_faiss(query_vector, top_k)
        elif self.vector_store_type == "qdrant" and self.qdrant_client is not None:
            return await self._search_qdrant(query_vector, top_k, tenant_id)
        elif self.vector_store_type == "memory":
            return self._search_memory(query_vector, top_k)
        else:
            logger.warning("Vector store not initialized, returning empty results")
            return []
    
    def _search_faiss(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """FAISS搜索"""
        try:
            query_vec = np.array([query_vector], dtype=np.float32)
            distances, indices = self.index.search(query_vec, top_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:  # FAISS返回-1表示无效结果
                    continue
                
                chunk_id = self.id_to_chunk.get(idx)
                if chunk_id:
                    # 将距离转换为相似度分数（FAISS使用L2距离，越小越相似）
                    # 转换为相似度：1 / (1 + distance)
                    similarity_score = 1.0 / (1.0 + float(distance))
                    results.append({
                        'id': chunk_id,
                        'score': similarity_score,
                        'rank': i + 1
                    })
            
            return results
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    async def _search_qdrant(self, query_vector: List[float], top_k: int, tenant_id: int) -> List[Dict[str, Any]]:
        """Qdrant搜索"""
        try:
            # 构建过滤条件（按tenant_id过滤）
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id)
                    )
                ]
            )
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_condition
            )
            
            results = []
            for i, result in enumerate(search_results):
                chunk_id = result.payload.get('chunk_id')
                if chunk_id:
                    results.append({
                        'id': chunk_id,
                        'score': float(result.score),
                        'rank': i + 1
                    })
            
            return results
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []
    
    def _search_memory(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """内存搜索（降级方案）"""
        if not self.vectors:
            return []
        
        query_vec = np.array(query_vector)
        scores = []
        
        for vector_id, vector in self.vectors.items():
            vec = np.array(vector)
            # 计算余弦相似度
            similarity = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
            chunk_id = self.id_to_chunk.get(vector_id)
            if chunk_id:
                scores.append({
                    'id': chunk_id,
                    'score': float(similarity),
                    'rank': 0  # 稍后排序
                })
        
        # 按分数降序排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 更新排名
        for i, result in enumerate(scores[:top_k]):
            result['rank'] = i + 1
        
        return scores[:top_k]
    
    def add_vectors(
        self,
        vectors: List[List[float]],
        chunk_ids: List[int],
        vector_ids: Optional[List[str]] = None
    ):
        """
        添加向量到索引（仅用于FAISS和内存存储）
        
        Args:
            vectors: 向量列表
            chunk_ids: 对应的chunk ID列表
            vector_ids: 可选的向量ID列表
        """
        if self.vector_store_type == "faiss":
            self._add_to_faiss(vectors, chunk_ids, vector_ids)
        elif self.vector_store_type == "memory":
            self._add_to_memory(vectors, chunk_ids, vector_ids)
    
    def _add_to_faiss(self, vectors: List[List[float]], chunk_ids: List[int], vector_ids: Optional[List[str]]):
        """添加到FAISS索引"""
        if not FAISS_AVAILABLE:
            return
        
        try:
            vectors_array = np.array(vectors, dtype=np.float32)
            
            if self.index is None:
                # 创建新索引
                dimension = len(vectors[0])
                self.index = faiss.IndexFlatL2(dimension)
                logger.info(f"Created new FAISS index with dimension {dimension}")
            
            # 添加到索引
            current_size = self.index.ntotal
            self.index.add(vectors_array)
            
            # 更新ID映射
            for i, chunk_id in enumerate(chunk_ids):
                vector_idx = current_size + i
                vector_id = vector_ids[i] if vector_ids else str(vector_idx)
                self.id_to_chunk[vector_idx] = chunk_id
                self.chunk_to_id[chunk_id] = vector_idx
            
            logger.info(f"Added {len(vectors)} vectors to FAISS index")
        except Exception as e:
            logger.error(f"Failed to add vectors to FAISS: {e}")
    
    def _add_to_memory(self, vectors: List[List[float]], chunk_ids: List[int], vector_ids: Optional[List[str]]):
        """添加到内存存储"""
        for i, (vector, chunk_id) in enumerate(zip(vectors, chunk_ids)):
            vector_id = vector_ids[i] if vector_ids else f"mem_{chunk_id}"
            self.vectors[vector_id] = vector
            self.id_to_chunk[vector_id] = chunk_id
            self.chunk_to_id[chunk_id] = vector_id
        
        if self.vector_dim is None and vectors:
            self.vector_dim = len(vectors[0])
    
    def save_index(self):
        """保存索引（仅用于FAISS）"""
        if self.vector_store_type == "faiss" and self.index is not None:
            try:
                index_file = Path(self.index_path)
                index_file.parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(self.index, str(index_file))
                
                # 保存ID映射
                mapping_file = index_file.parent / f"{index_file.name}.mapping"
                with open(mapping_file, 'wb') as f:
                    pickle.dump(self.id_to_chunk, f)
                
                logger.info(f"Saved FAISS index to {index_file}")
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
