"""
会话状态管理：按 session_id 存储当前对话的已填槽位与目标意图
使用内存 Dict，可后续替换为 Redis
"""
import asyncio
from typing import Dict, Any, Optional, List

# session_id -> { "current_slots": Dict[str, str], "target_intent": str }
_session_store: Dict[str, Dict[str, Any]] = {}
_session_lock = asyncio.Lock()


async def get_state(session_id: Optional[str]) -> Dict[str, Any]:
    """
    获取会话状态。若 session_id 为空或不存在，返回空状态。
    Returns:
        { "current_slots": Dict[str, str], "target_intent": Optional[str] }
    """
    if not session_id:
        return {"current_slots": {}, "target_intent": None}
    async with _session_lock:
        state = _session_store.get(session_id)
        if not state:
            return {"current_slots": {}, "target_intent": None}
        return {
            "current_slots": dict(state.get("current_slots") or {}),
            "target_intent": state.get("target_intent"),
        }


async def set_state(
    session_id: str,
    current_slots: Dict[str, str],
    target_intent: Optional[str] = None,
) -> None:
    """更新会话的已填槽位与目标意图。"""
    async with _session_lock:
        _session_store.setdefault(session_id, {})
        _session_store[session_id]["current_slots"] = dict(current_slots)
        if target_intent is not None:
            _session_store[session_id]["target_intent"] = target_intent


async def merge_slots(
    session_id: Optional[str],
    new_slots: Dict[str, str],
    target_intent: Optional[str],
) -> Dict[str, str]:
    """
    合并会话中已有槽位与本次新抽取的槽位（新值覆盖旧值），并写回会话状态。
    Returns:
        合并后的 current_slots。
    """
    state = await get_state(session_id)
    merged = dict(state.get("current_slots") or {})
    for k, v in new_slots.items():
        if v:
            merged[k] = v
    if session_id:
        await set_state(session_id, merged, target_intent)
    return merged


async def clear_session(session_id: Optional[str]) -> None:
    """清除会话状态（可选，用于对话结束或重置）。"""
    if not session_id:
        return
    async with _session_lock:
        _session_store.pop(session_id, None)
