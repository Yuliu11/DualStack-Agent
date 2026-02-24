"""
LLM 客户端服务
支持 DeepSeek（主）和阿里云百炼（备）双模型，包含重试、熔断、降级等机制
"""
import asyncio
import time
import random
import re
from typing import Optional, List, Dict, Any, AsyncIterator, Union
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import httpx

from app.config.config import settings, yaml_config, prompt_config
from app.storage.db_manager import db_manager
from app.storage.models import ModelEvent
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"  # 正常状态
    OPEN = "open"  # 熔断状态
    HALF_OPEN = "half_open"  # 半开状态（尝试恢复）


@dataclass
class CircuitBreaker:
    """熔断器"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: CircuitState = CircuitState.CLOSED
    failure_threshold: int = 5
    recovery_timeout: int = 300  # 5分钟，单位秒
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def should_attempt(self) -> bool:
        """判断是否应该尝试调用"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # 检查是否超过恢复超时时间
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True


class LLMService:
    """LLM 服务类，支持双模型、重试、熔断、降级"""
    
    def __init__(self):
        """初始化 LLM 服务"""
        # 加载配置
        self.retry_config = yaml_config.get("retry", {})
        self.circuit_config = yaml_config.get("circuit_breaker", {})
        self.models_config = yaml_config.get("models", {})
        self.qa_policy = yaml_config.get("qa_policy", {})
        
        # 初始化 DeepSeek 客户端（主模型）
        # 确保 base_url 不包含 /v1（DeepSeek API 会自动添加）
        deepseek_base_url = settings.deepseek_base_url
        if deepseek_base_url.endswith("/v1"):
            deepseek_base_url = deepseek_base_url.rstrip("/v1").rstrip("/")
            logger.warning(f"DeepSeek base_url 包含 /v1，已自动修正为: {deepseek_base_url}")
        
        self.deepseek_client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=deepseek_base_url
        )
        self.deepseek_model = self.models_config.get("deepseek", {}).get("chat_model", "deepseek-chat")
        
        # 初始化阿里云百炼客户端（备用模型）
        # 注意：阿里云百炼的 OpenAI 兼容接口使用兼容模式路径
        dashscope_base_url = settings.dashscope_base_url
        # 阿里云百炼的 OpenAI 兼容接口路径：/compatible-mode/v1
        if "/compatible-mode" not in dashscope_base_url:
            if dashscope_base_url.endswith("/v1"):
                dashscope_base_url = dashscope_base_url.replace("/v1", "/compatible-mode/v1")
            elif not dashscope_base_url.endswith("/compatible-mode/v1"):
                dashscope_base_url = f"{dashscope_base_url.rstrip('/')}/compatible-mode/v1"
        
        self.dashscope_client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=dashscope_base_url
        )
        self.dashscope_model = self.models_config.get("dashscope", {}).get("chat_model", "qwen-turbo")
        
        # 熔断器
        self.deepseek_circuit = CircuitBreaker(
            failure_threshold=self.circuit_config.get("failure_threshold", 5),
            recovery_timeout=self.circuit_config.get("recovery_timeout", 300)
        )
        
        # 租户ID（默认0）
        self.default_tenant_id = 0
        
        # 加载Prompt配置
        self._load_prompts()
    
    def _is_retryable_exception(self, exc: Exception) -> bool:
        """判断异常是否可重试"""
        if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException)):
            return True
        if isinstance(exc, (ConnectionError, httpx.ConnectError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            # 429 限流错误
            if exc.response.status_code == 429:
                return True
            # 5xx 服务器错误
            if 500 <= exc.response.status_code < 600:
                return True
        # OpenAI API 错误
        if hasattr(exc, "status_code"):
            if exc.status_code == 429:
                return True
            if 500 <= exc.status_code < 600:
                return True
        return False
    
    def _load_prompts(self):
        """加载Prompt配置（启动时自动调用）"""
        try:
            # 预加载prompt配置，验证文件是否存在
            prompt_config.load()
            logger.info("Prompt配置加载成功")
        except Exception as e:
            logger.warning(f"加载Prompt配置失败: {e}，将使用默认提示词")
    
    def load_prompt(self, key: str, **kwargs) -> str:
        """
        加载并格式化Prompt模板
        
        Args:
            key: Prompt键，例如 "rag_answer.system_template"
            **kwargs: 格式化参数
        
        Returns:
            格式化后的Prompt字符串
        """
        try:
            return prompt_config.format(key, **kwargs)
        except Exception as e:
            logger.error(f"加载Prompt失败: {key}, 错误: {e}")
            # 降级：返回默认提示词
            if key == "rag_answer.system_template":
                context = kwargs.get("context", "")
                return f"""你是一个专业的文档问答助手。请根据提供的上下文回答用户的问题。

**重要要求：**
1. 必须严格根据提供的上下文回答，严禁自我发挥或添加上下文之外的信息
2. 每个关键结论必须标注来源，格式为：[Chunk ID]
3. 如果上下文中没有相关信息，请明确说明"根据提供的上下文，无法找到相关信息"
4. 不要编造或推测信息

上下文：
{context}"""
            else:
                raise
    
    async def _log_model_event(
        self,
        model_name: str,
        event_type: str,
        http_status: Optional[int] = None,
        error_code: Optional[str] = None,
        error_msg: Optional[str] = None,
        retry_count: int = 0,
        fallback_to: Optional[str] = None,
        request_id: Optional[int] = None
    ):
        """记录模型事件到数据库"""
        try:
            async with db_manager.get_session() as session:
                event = ModelEvent(
                    tenant_id=self.default_tenant_id,
                    request_id=request_id,
                    model_name=model_name,
                    event_type=event_type,
                    http_status=http_status,
                    error_code=error_code,
                    error_msg=str(error_msg)[:65535] if error_msg else None,  # TEXT 字段限制
                    retry_count=retry_count,
                    fallback_to=fallback_to
                )
                session.add(event)
                await session.commit()
        except Exception as e:
            # 记录事件失败不应该影响主流程，只打印日志
            print(f"Failed to log model event: {e}")
    
    async def _call_with_retry(
        self,
        func,
        model_name: str,
        *args,
        **kwargs
    ):
        """带重试的调用。当 func 返回 AsyncIterator（stream=True）时，直接返回该迭代器，不当作 ChatCompletion 处理。"""
        max_attempts = self.retry_config.get("max_attempts", 3)
        wait_multiplier = self.retry_config.get("wait_exponential_multiplier", 1)
        wait_max = self.retry_config.get("wait_exponential_max", 10)
        stop_delay = self.retry_config.get("stop_after_delay", 30)
        
        attempt = 0
        last_exc = None
        start_time = time.time()
        
        while attempt < max_attempts:
            attempt += 1
            try:
                # 彻底删除 request_id，确保不会传递给 SDK
                kwargs.pop('request_id', None)
                result = await func(*args, **kwargs)
                # 流式时 result 为 AsyncIterator，直接返回，避免被当作 ChatCompletion 做任何访问
                return result
            except Exception as e:
                last_exc = e
                
                # 检查是否可重试
                if not self._is_retryable_exception(e):
                    raise
                
                # 检查是否超过总延迟时间
                if time.time() - start_time > stop_delay:
                    await self._log_model_event(
                        model_name=model_name,
                        event_type="timeout",
                        error_code=type(e).__name__,
                        error_msg=f"Retry timeout after {stop_delay}s",
                        retry_count=attempt,
                        request_id=kwargs.get("request_id")
                    )
                    raise
                
                # 记录重试事件
                http_status = getattr(e, "status_code", None)
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    http_status = e.response.status_code
                
                await self._log_model_event(
                    model_name=model_name,
                    event_type="retry",
                    http_status=http_status,
                    error_code=type(e).__name__,
                    error_msg=str(e),
                    retry_count=attempt,
                    request_id=kwargs.get("request_id")
                )
                
                # 如果还有重试机会，等待后重试
                if attempt < max_attempts:
                    wait_time = min(wait_multiplier * (2 ** (attempt - 1)), wait_max)
                    # 添加随机抖动（0-1秒）
                    wait_time += random.uniform(0, 1)
                    await asyncio.sleep(wait_time)
                else:
                    # 最后一次尝试失败
                    raise
        
        if last_exc:
            raise last_exc
        raise Exception("Retry exhausted")
    
    async def _call_deepseek(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        request_id: Optional[int] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator]:
        """调用 DeepSeek API。stream=True 时返回 AsyncIterator，否则返回 ChatCompletion。"""
        async def _call():
            response = await self.deepseek_client.chat.completions.create(
                model=self.deepseek_model,
                messages=messages,
                stream=stream,
                **kwargs
            )
            return response
        
        return await self._call_with_retry(
            _call,
            "deepseek-chat",
            request_id=request_id
        )
    
    async def _call_dashscope(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        request_id: Optional[int] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator]:
        """调用阿里云百炼 API。stream=True 时返回 AsyncIterator，否则返回 ChatCompletion。"""
        async def _call():
            # DashScope SDK 只支持标准 OpenAI 兼容参数
            # 彻底删除所有非标准参数，确保不会传递给 SDK
            # 只保留 SDK 支持的参数：temperature, max_tokens, top_p, frequency_penalty, presence_penalty, etc.
            unsupported_params = {
                'request_id',  # 非 SDK 参数
                'extra_headers',  # OpenAI SDK 扩展参数
                'extra_query',  # OpenAI SDK 扩展参数
                'extra_body',  # OpenAI SDK 扩展参数
            }
            clean_kwargs = {
                k: v for k, v in kwargs.items() 
                if k not in unsupported_params
            }
            # 双重保险：再次确保 request_id 被移除
            clean_kwargs.pop('request_id', None)
            
            response = await self.dashscope_client.chat.completions.create(
                model=self.dashscope_model,
                messages=messages,
                stream=stream,
                **clean_kwargs
            )
            return response
        
        return await self._call_with_retry(
            _call,
            "dashscope-qwen",
            request_id=request_id
        )
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        request_id: Optional[int] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator]:
        """
        聊天补全方法，支持主备模型切换。

        Args:
            messages: 消息列表
            stream: 是否流式返回
            request_id: 请求ID（用于关联 model_events）
            **kwargs: 其他 OpenAI API 参数

        Returns:
            stream=False 时返回 ChatCompletion；stream=True 时返回 AsyncIterator（可直接 async for 迭代）。
        """
        # 优先尝试主模型（DeepSeek）
        if self.deepseek_circuit.should_attempt():
            try:
                response = await self._call_deepseek(
                    messages,
                    stream=stream,
                    request_id=request_id,
                    **kwargs
                )
                # 获取到响应（非流时为 ChatCompletion，流时为 AsyncIterator）即视为成功，重置熔断器
                self.deepseek_circuit.record_success()
                return response
            except Exception as e:
                # 记录失败
                self.deepseek_circuit.record_failure()
                
                # 记录异常事件
                http_status = getattr(e, "status_code", None)
                if hasattr(e, "response") and hasattr(e.response, "status_code"):
                    http_status = e.response.status_code
                
                event_type = "timeout" if isinstance(e, (asyncio.TimeoutError, httpx.TimeoutException)) else \
                           "rate_limit" if (http_status == 429) else \
                           "5xx" if (http_status and 500 <= http_status < 600) else \
                           "parse_error" if hasattr(e, "parse_error") else \
                           "fallback"
                
                await self._log_model_event(
                    model_name="deepseek-chat",
                    event_type=event_type,
                    http_status=http_status,
                    error_code=type(e).__name__,
                    error_msg=str(e),
                    retry_count=0,
                    fallback_to="dashscope-qwen",
                    request_id=request_id
                )
                
                # 如果主模型失败，降级到备用模型
                if self.deepseek_circuit.state == CircuitState.OPEN:
                    # 熔断状态，直接使用备用模型
                    pass
                else:
                    # 尝试备用模型
                    pass
        
        # 使用备用模型（阿里云百炼）
        try:
            await self._log_model_event(
                model_name="dashscope-qwen",
                event_type="fallback",
                fallback_to=None,
                request_id=request_id
            )
            # 调用 DashScope 时，不传递 request_id 给 SDK（仅用于日志记录）
            response = await self._call_dashscope(
                messages,
                stream=stream,
                request_id=request_id,  # 仅用于日志记录，不会传递给 SDK
                **kwargs
            )
            # 流式时 response 为 AsyncIterator，原样返回供上层 async for 消费
            return response
        except Exception as e:
            # 备用模型也失败
            http_status = getattr(e, "status_code", None)
            if hasattr(e, "response") and hasattr(e.response, "status_code"):
                http_status = e.response.status_code
            
            await self._log_model_event(
                model_name="dashscope-qwen",
                event_type="5xx" if (http_status and 500 <= http_status < 600) else "parse_error",
                http_status=http_status,
                error_code=type(e).__name__,
                error_msg=str(e),
                request_id=request_id
            )
            raise

    async def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        request_id: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式生成答案：与 generate_answer 使用相同上下文与提示，但 stream=True，
        正确消费 AsyncOpenAI 返回的 stream 对象并逐块 yield 内容。
        """
        context_text = "\n\n".join([
            f"[Chunk {chunk['id']}]\n{chunk['content']}"
            for chunk in context_chunks
        ])
        system_prompt = self.load_prompt(
            "rag_answer.system_template",
            context=context_text
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        raw = await self.chat_completion(
            messages=messages,
            stream=True,
            request_id=request_id,
            **kwargs
        )
        if raw is None:
            return
        # 确保得到的是可异步迭代的对象（AsyncOpenAI stream 对象）
        stream = raw
        if not hasattr(stream, "__aiter__"):
            return
        async for chunk in stream:
            if chunk is None:
                continue
            choices = getattr(chunk, "choices", None)
            if not choices or len(choices) == 0:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content is None and isinstance(delta, dict):
                content = delta.get("content")
            if not content:
                continue
            await asyncio.sleep(0)
            yield content
    
    def _extract_chunk_ids(self, text: str) -> List[int]:
        """从文本中提取 Chunk ID（格式：[Chunk ID] 或 [Chunk 123]）"""
        pattern = r'\[Chunk\s+(\d+)\]'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [int(match) for match in matches]
    
    def _calculate_confidence(self, answer: str, context_chunks: List[Dict[str, Any]]) -> float:
        """
        计算答案的置信度
        
        基于以下因素：
        1. 是否包含 Chunk ID 引用
        2. 答案长度（过短可能表示不完整）
        3. 是否包含明确的拒绝语句
        """
        confidence = 0.5  # 基础置信度
        
        # 检查是否包含 Chunk ID 引用
        chunk_ids = self._extract_chunk_ids(answer)
        if chunk_ids:
            confidence += 0.3
            # 引用的 chunk 数量越多，置信度越高（但有限制）
            confidence += min(len(chunk_ids) * 0.05, 0.15)
        else:
            confidence -= 0.2
        
        # 检查答案长度
        if len(answer.strip()) < 10:
            confidence -= 0.3  # 答案过短
        elif len(answer.strip()) > 50:
            confidence += 0.1  # 答案有足够内容
        
        # 检查是否包含拒绝语句
        refuse_keywords = ["无法", "不知道", "没有找到", "未找到", "不清楚", "无法确定"]
        if any(keyword in answer for keyword in refuse_keywords):
            confidence -= 0.2
        
        # 限制在 [0, 1] 范围内
        return max(0.0, min(1.0, confidence))
    
    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        request_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成答案，强制引用约束
        
        Args:
            query: 用户查询
            context_chunks: 上下文块列表，每个块应包含 {'id': int, 'content': str, ...}
            request_id: 请求ID
            **kwargs: 其他参数
        
        Returns:
            {
                'answer': str,  # 答案文本
                'confidence': float,  # 置信度 [0, 1]
                'chunk_ids': List[int],  # 引用的 Chunk ID 列表
                'is_refused': bool,  # 是否拒答
                'model_name': str  # 使用的模型名称
            }
        """
        # 构建上下文
        context_text = "\n\n".join([
            f"[Chunk {chunk['id']}]\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        # 构建系统提示（从配置文件加载）
        system_prompt = self.load_prompt(
            "rag_answer.system_template",
            context=context_text
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # 调用 LLM
        try:
            response = await self.chat_completion(
                messages=messages,
                stream=False,
                request_id=request_id,
                **kwargs
            )
            
            # 提取答案
            answer = response.choices[0].message.content if response.choices else ""
            model_name = response.model if hasattr(response, "model") else "unknown"
            
            # 提取引用的 Chunk ID
            chunk_ids = self._extract_chunk_ids(answer)
            
            # 计算置信度
            confidence = self._calculate_confidence(answer, context_chunks)
            
            # 判断是否拒答
            min_confidence = self.qa_policy.get("min_confidence", 0.6)
            is_refused = False
            
            if not answer or answer.strip() == "":
                is_refused = True
            elif self.qa_policy.get("refuse_if_low_conf", True) and confidence < min_confidence:
                is_refused = True
            elif self.qa_policy.get("cite_required", True) and not chunk_ids:
                # 要求引用但未找到引用
                is_refused = True
            
            return {
                "answer": answer,
                "confidence": confidence,
                "chunk_ids": chunk_ids,
                "is_refused": is_refused,
                "model_name": model_name
            }
        except Exception as e:
            # 生成答案失败
            await self._log_model_event(
                model_name="unknown",
                event_type="parse_error",
                error_code=type(e).__name__,
                error_msg=str(e),
                request_id=request_id
            )
            raise
