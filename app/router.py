"""
意图识别模块
根据用户查询将其归类为 FACT / CALC / SUMMARY，用于后续检索与生成策略路由
"""
import logging
from typing import Literal

from app.llm import llm_service

logger = logging.getLogger(__name__)

IntentType = Literal["FACT", "CALC", "SUMMARY"]

SYSTEM_PROMPT = """你是一个查询意图分类器。根据用户的问题，只输出以下三个单词之一，不要带任何解释、标点或换行。

- FACT：单点事实类问题（如：某概念是什么、谁、何时、哪里等）
- CALC：涉及数据计算或对比的问题（如：增长、同期、比例、谁更多、差值等）
- SUMMARY：全文总结类问题（如：总结、奋斗、核心观点、主要内容等）

分类规则简要提示：
- 若问题包含“增长、同期、比例、谁更多、差值”等，输出 CALC
- 若问题包含“总结、奋斗、核心观点、主要内容”等，输出 SUMMARY
- 其他情况输出 FACT

只输出一个单词：FACT、CALC 或 SUMMARY。"""


class IntentRouter:
    """意图路由器：将用户查询归类为 FACT / CALC / SUMMARY"""

    VALID_INTENTS = frozenset({"FACT", "CALC", "SUMMARY"})

    def __init__(self, llm_service_instance=None):
        """
        Args:
            llm_service_instance: 可选，LLM 服务实例；不传则使用全局 llm_service
        """
        self._llm = llm_service_instance if llm_service_instance is not None else llm_service

    def _parse_llm_output(self, raw: str) -> IntentType:
        """解析 LLM 输出，无效时返回 FACT。"""
        if not raw or not isinstance(raw, str):
            return "FACT"
        text = raw.strip().upper()
        # 只取第一个“词”（允许前后有空格，中间无空格）
        first_word = text.split()[0] if text.split() else ""
        if first_word in self.VALID_INTENTS:
            return first_word  # type: ignore
        # 兼容可能带标点的情况，如 "FACT." / "CALC。"
        for intent in self.VALID_INTENTS:
            if text.startswith(intent) or intent in text.split():
                return intent  # type: ignore
        return "FACT"

    # 预设关键词：命中则直接返回对应意图，避免“查询报表”等被误判为 FACT
    SUMMARY_KEYWORDS = ("报表", "总结", "汇总", "概括", "核心观点", "主要内容", "奋斗", "查询报表", "财务报表")
    CALC_KEYWORDS = ("增长", "同期", "比例", "谁更多", "差值", "计算", "同比", "环比")

    async def get_intent(self, query: str) -> IntentType:
        """
        根据用户查询获取意图类型。
        先走预设关键词匹配（报表/总结->SUMMARY，计算/增长->CALC），再走 LLM。
        """
        if not (query and query.strip()):
            return "FACT"
        q = query.strip()
        for kw in self.SUMMARY_KEYWORDS:
            if kw in q:
                logger.debug("意图预设命中 SUMMARY: 关键词 %s", kw)
                return "SUMMARY"
        for kw in self.CALC_KEYWORDS:
            if kw in q:
                logger.debug("意图预设命中 CALC: 关键词 %s", kw)
                return "CALC"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query.strip()},
        ]

        try:
            response = await self._llm.chat_completion(
                messages=messages,
                stream=False,
                temperature=0.0,
                max_tokens=16,
            )
            raw = (
                response.choices[0].message.content
                if response.choices and response.choices[0].message
                else ""
            )
            intent = self._parse_llm_output(raw)
            if raw and intent == "FACT" and raw.strip().upper() not in self.VALID_INTENTS:
                logger.debug("意图解析兜底为 FACT，原始输出: %s", raw[:200])
            return intent
        except Exception as e:
            logger.warning("意图识别 LLM 调用失败，兜底为 FACT: %s", e)
            return "FACT"


# 单例，便于全局使用
intent_router = IntentRouter()

__all__ = ["IntentRouter", "IntentType", "intent_router"]
