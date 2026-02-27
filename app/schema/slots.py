"""
槽位 Schema 与解析配置
按意图定义所需槽位（name、required），供 SlotExtractor 与 RAG 流程使用
"""
from typing import Dict, List, Any

# 意图 -> 该意图所需槽位列表；每个槽位 name, required, 以及补槽时的追问 question
intent_slots: Dict[str, List[Dict[str, Any]]] = {
    "CALC": [
        {"name": "product", "required": True, "question": "请问您查的是哪个产品？"},
        {"name": "date", "required": False, "question": "请问您要查询哪个时间或年份？"},
    ],
    "SUMMARY": [
        {"name": "year", "required": True, "question": "请问您要查询哪一年（年份）？"},
        {"name": "dept", "required": True, "question": "请问您要查询哪个部门？"},
        {"name": "date", "required": False, "question": "请问您要查询哪个时间或年份？"},
        {"name": "department", "required": False, "question": "请问您要查询哪个部门？"},
        {"name": "product", "required": False, "question": "请问您要总结的是哪个产品？"},
    ],
    "FACT": [],  # 事实类可不配置槽位，直接走检索
}


def get_slots_for_intent(intent: str) -> List[Dict[str, Any]]:
    """
    根据意图查询所需槽位配置。
    若意图未在 intent_slots 中配置，返回空列表（不拦流程）。
    """
    return intent_slots.get(intent, [])


def get_required_slot_names(intent: str) -> List[str]:
    """返回该意图下必填槽位名称列表。"""
    slots = get_slots_for_intent(intent)
    return [s["name"] for s in slots if s.get("required")]


def get_question_for_slot(intent: str, slot_name: str) -> str:
    """返回该意图下某槽位在缺填时的追问文案；无则返回默认句。"""
    slots = get_slots_for_intent(intent)
    for s in slots:
        if s.get("name") == slot_name and s.get("question"):
            return s["question"]
    return f"请提供「{slot_name}」相关信息。"
