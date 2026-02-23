"""
LLM 模块
导出 LLMService 单例
"""
from app.llm.client import LLMService

# 创建全局单例
llm_service = LLMService()

__all__ = ["llm_service", "LLMService"]