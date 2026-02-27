"""
查询扩写/改写模块
实现 Query Rewriter 功能，包括上下文补全、关键词扩展、多维度重写；
以及 SlotExtractor 槽位抽取（LLM 解析用户输入为 JSON 槽位）。
"""
import logging
import json
from typing import Dict, Any, Optional, List
from app.llm import llm_service
from app.config.config import yaml_config, prompt_config

# 配置日志
logger = logging.getLogger(__name__)


# ---------- 槽位抽取 ----------
SLOT_EXTRACT_SYSTEM = """你是一个槽位抽取器。根据用户当前输入，提取指定槽位的值。
只输出一个 JSON 对象，键为槽位名，值为从用户输入中识别出的字符串（无则省略该键或值为空字符串）。
不要输出任何解释、换行或 markdown 标记，仅输出 JSON。"""


class SlotExtractor:
    """使用 LLM 将用户当前输入解析为 JSON 格式的槽位信息。"""

    def __init__(self, llm_service_instance=None):
        self._llm = llm_service_instance if llm_service_instance is not None else llm_service

    async def extract(self, query: str, slot_names: List[str]) -> Dict[str, str]:
        """
        从用户输入中抽取指定槽位的值。

        Args:
            query: 用户当前输入
            slot_names: 需要抽取的槽位名称列表，如 ["product", "date"]

        Returns:
            槽位名 -> 抽取值的字典；未识别或空的槽位不包含或值为空字符串。
        """
        if not slot_names:
            return {}
        slots_str = "、".join(slot_names)
        user_prompt = f"槽位名列表：{slots_str}\n\n用户输入：{query.strip()}"
        messages = [
            {"role": "system", "content": SLOT_EXTRACT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = await self._llm.chat_completion(
                messages=messages,
                stream=False,
                temperature=0.0,
                max_tokens=256,
            )
            raw = (
                response.choices[0].message.content
                if response.choices and response.choices[0].message
                else ""
            )
            raw = raw.strip()
            if not raw:
                return {}
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.lower().startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            result = {}
            for k in slot_names:
                v = data.get(k)
                if v is not None and str(v).strip():
                    result[k] = str(v).strip()
            return result
        except Exception as e:
            logger.warning("[SlotExtractor] 解析失败: %s", e)
            return {}


class QueryRewriter:
    """查询扩写/改写器"""
    
    def __init__(self):
        """初始化查询改写器"""
        self.enabled = yaml_config.get("retrieval", {}).get("query_rewrite_enabled", True)
        self.max_retries = 2
        self.timeout = 10  # 秒
    
    async def rewrite_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        tenant_id: int = 0
    ) -> Dict[str, Any]:
        """
        扩写/改写查询
        
        Args:
            query: 原始查询
            conversation_history: 对话历史，格式为 [{"role": "user", "content": "..."}, ...]
            tenant_id: 租户ID
        
        Returns:
            {
                'original_query': str,  # 原始查询
                'rewritten_queries': List[str],  # 改写后的查询列表（最多3个）
                'primary_query': str,  # 主要查询（用于显示）
                'rewrite_type': str,  # 改写类型：'full' | 'degraded' | 'original'
                'rewrite_details': dict  # 改写详情
            }
        """
        if not self.enabled:
            logger.debug(f"[Query Rewrite] 功能已禁用，使用原始查询: {query}")
            return {
                'original_query': query,
                'rewritten_queries': [query],
                'primary_query': query,
                'rewrite_type': 'disabled',
                'rewrite_details': {},
                'extracted_years': [],
            }
        
        try:
            # 构建改写提示
            rewrite_prompt = self._build_rewrite_prompt(query, conversation_history)
            
            # 调用 LLM 进行改写
            logger.info(f"[Query Rewrite] 开始改写查询: {query[:50]}...")
            
            # 获取用户提示词模板（从配置文件加载）
            try:
                user_prompt = prompt_config.format(
                    "query_rewrite.user_template",
                    query=query
                )
            except Exception as e:
                logger.warning(f"加载查询改写用户Prompt失败: {e}，使用默认提示词")
                user_prompt = f"请改写以下查询：\n{query}"
            
            messages = [
                {
                    "role": "system",
                    "content": rewrite_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
            
            # 调用 LLM（使用 chat_completion）
            # 注意：不传递 request_id，避免参数冲突
            response = await llm_service.chat_completion(
                messages=messages,
                stream=False,
                temperature=0.3,  # 较低温度，保持一致性
                max_tokens=500
                # 不传递 request_id，避免 DashScope SDK 参数冲突
            )
            
            # 提取响应内容
            if hasattr(response, 'choices') and len(response.choices) > 0:
                rewritten_text = response.choices[0].message.content.strip()
            else:
                rewritten_text = str(response).strip()
            
            if not rewritten_text:
                raise ValueError("LLM 返回空结果")
            
            # 解析 LLM 返回的 JSON
            rewrite_result = self._parse_rewrite_result(rewritten_text, query)
            
            # 构建主要查询（用于显示）
            primary_query = rewrite_result.get('primary_query', query)
            
            # 构建改写后的查询列表（最多3个）
            rewritten_queries = rewrite_result.get('rewritten_queries', [query])
            if not rewritten_queries:
                rewritten_queries = [query]
            
            # 确保至少有3个查询（如果不足，补充原始查询）
            if len(rewritten_queries) < 3:
                # 如果只有1-2个，补充原始查询或变体
                while len(rewritten_queries) < 3:
                    if query not in rewritten_queries:
                        rewritten_queries.append(query)
                    else:
                        # 如果原始查询已在列表中，添加一个变体
                        rewritten_queries.append(f"{query} 详细信息")
                        break
            else:
                # 如果超过3个，只取前3个
                rewritten_queries = rewritten_queries[:3]
            
            # 如果只有一个查询，确保至少使用原始查询
            if len(rewritten_queries) == 0:
                rewritten_queries = [query]
            
            # 日志打印对比
            logger.info(
                f"[Query Rewrite] \"{query}\" -> \"{primary_query}\" "
                f"(生成 {len(rewritten_queries)} 个查询)"
            )
            
            details = rewrite_result.get('details', {})
            details['extracted_years'] = rewrite_result.get('extracted_years', [])
            return {
                'original_query': query,
                'rewritten_queries': rewritten_queries,
                'primary_query': primary_query,
                'rewrite_type': 'full',
                'rewrite_details': details,
                'extracted_years': rewrite_result.get('extracted_years', []),
            }
            
        except Exception as e:
            # 异常处理：降级到原始查询
            logger.warning(
                f"[Query Rewrite] 改写失败，降级使用原始查询: {query}, "
                f"错误: {str(e)}"
            )
            
            return {
                'original_query': query,
                'rewritten_queries': [query],
                'primary_query': query,
                'rewrite_type': 'degraded',
                'rewrite_details': {'error': str(e)},
                'extracted_years': [],
            }
    
    def _build_rewrite_prompt(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """构建改写提示（从配置文件加载）"""
        
        history_context = ""
        if conversation_history:
            # 提取最近3轮对话作为上下文
            recent_history = conversation_history[-6:]  # 最多3轮（每轮2条消息）
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_history
            ])
            history_context = f"\n\n对话历史：\n{history_text}"
        
        try:
            # 从配置文件加载prompt模板
            prompt = prompt_config.format(
                "query_rewrite.system_template",
                history_context=history_context
            )
            return prompt
        except Exception as e:
            logger.warning(f"加载查询改写Prompt失败: {e}，使用默认提示词")
            # 降级：返回默认提示词
            return f"""你是一个专业的查询改写助手。请分析用户的查询，并进行以下三个任务的改写：

**任务A：上下文补全**
如果查询中包含代词（如"它"、"该公司"、"这份报告"、"那个"等），请根据对话历史补全为具体的实体名称。
如果没有对话历史或无法确定实体，保持原样。

**任务B：关键词扩展**
将模糊词汇扩展为更具体的行业术语和同义词。
例如：
- "业务" -> "主营业务、经营范围、核心产品、业务模式"
- "财务" -> "财务状况、财务报表、营收、利润、资产"
- "风险" -> "风险因素、风险提示、经营风险、市场风险"

**任务C：多维度重写**
将一个问题扩展为2-3个从不同角度切入的搜索语句，确保覆盖问题的多个维度。
例如：
- 原查询："它的业务是什么？"
- 改写1："大树科技的主营业务是什么？"
- 改写2："大树科技的经营范围和核心产品有哪些？"
- 改写3："大树科技的业务模式和收入来源是什么？"

**任务D：时间维度提取**
从用户问题中提取时间/年份（如 2023、2024、去年、今年等），用于检索时过滤文档。在输出的 JSON 中增加 "extracted_years" 数组，如 [2023, 2024]；若无法识别或问题与年份无关则返回空数组 []。

**输出格式（必须是有效的JSON）：**
{{
    "primary_query": "改写后的主要查询（用于显示）",
    "rewritten_queries": ["查询1", "查询2", "查询3"],
    "extracted_years": [2023, 2024],
    "details": {{
        "context_completion": "上下文补全说明",
        "keyword_expansion": "关键词扩展说明",
        "multi_dimension": "多维度重写说明",
        "time_dimension": "时间维度提取说明"
    }}
}}

**要求：**
1. 必须返回有效的JSON格式
2. rewritten_queries 数组应包含2-3个不同角度的查询
3. extracted_years 为数字数组，仅包含从问题中识别出的年份
4. 如果无法改写，至少返回原始查询
5. 保持查询的语义完整性{history_context}"""
    
    def _parse_rewrite_result(self, llm_output: str, original_query: str) -> Dict[str, Any]:
        """解析 LLM 返回的改写结果"""
        try:
            # 尝试提取 JSON（可能包含 markdown 代码块）
            json_text = llm_output.strip()
            
            # 移除可能的 markdown 代码块标记
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            elif json_text.startswith("```"):
                json_text = json_text[3:]
            
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            json_text = json_text.strip()
            
            # 解析 JSON
            result = json.loads(json_text)
            
            # 验证必需字段
            if 'primary_query' not in result:
                result['primary_query'] = original_query
            
            if 'rewritten_queries' not in result or not result['rewritten_queries']:
                result['rewritten_queries'] = [result.get('primary_query', original_query)]
            if 'extracted_years' not in result or not isinstance(result.get('extracted_years'), list):
                result['extracted_years'] = []
            else:
                result['extracted_years'] = [int(y) for y in result['extracted_years'] if isinstance(y, (int, float)) or (isinstance(y, str) and y.isdigit())]
            if 'details' not in result:
                result['details'] = {}
            result['details']['extracted_years'] = result.get('extracted_years', [])
            return result
            
        except json.JSONDecodeError as e:
            logger.warning(f"[Query Rewrite] JSON 解析失败: {e}, 输出: {llm_output[:200]}")
            # 降级：尝试提取主要查询
            primary_query = self._extract_primary_query_from_text(llm_output, original_query)
            return {
                'primary_query': primary_query,
                'rewritten_queries': [primary_query],
                'details': {
                    'parse_error': str(e),
                    'raw_output': llm_output[:500]
                }
            }
        except Exception as e:
            logger.error(f"[Query Rewrite] 解析结果失败: {e}")
            return {
                'primary_query': original_query,
                'rewritten_queries': [original_query],
                'details': {'error': str(e)}
            }
    
    def _extract_primary_query_from_text(self, text: str, fallback: str) -> str:
        """从文本中提取主要查询（降级方案）"""
        # 尝试查找引号内的内容
        import re
        matches = re.findall(r'["\']([^"\']+)["\']', text)
        if matches:
            return matches[0]
        
        # 尝试查找"primary_query"后的内容
        match = re.search(r'primary[_\s]*query[:\s]+["\']?([^"\'\n]+)["\']?', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 如果文本看起来像是一个查询，直接使用
        if len(text) < 200 and '?' in text or '？' in text:
            return text.strip()
        
        return fallback


# 创建全局单例
query_rewriter = QueryRewriter()
slot_extractor = SlotExtractor()
