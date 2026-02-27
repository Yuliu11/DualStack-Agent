"""
应用层 Schema：槽位、意图等配置与数据结构
"""
from app.schema.slots import (
    intent_slots,
    get_slots_for_intent,
    get_required_slot_names,
    get_question_for_slot,
)

__all__ = [
    "intent_slots",
    "get_slots_for_intent",
    "get_required_slot_names",
    "get_question_for_slot",
]
