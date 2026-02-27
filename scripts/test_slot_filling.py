#!/usr/bin/env python3
"""
槽位填充多轮对话与流式输出自动化测试。

要求：
- 多轮对话：第一轮不完整指令断言引导「年份」或「部门」；第二轮同 session_id 补全后断言触发 RAG。
- 流式处理：解析 text/event-stream，控制台实时打印流式内容。
- 隔离性：不同 session_id 槽位互不干扰。

用法（需先启动 API 服务，如 uvicorn app.api.main:app）:
  python scripts/test_slot_filling.py
  python scripts/test_slot_filling.py --base-url http://localhost:8000
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

# 项目根目录加入 path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import requests
except ImportError:
    print("请安装 requests: pip install requests", file=sys.stderr)
    sys.exit(1)

DEFAULT_BASE_URL = "http://localhost:8000"


def parse_sse_stream(response, real_time_print=True):
    """
    解析 SSE 流，逐个 data 事件解析为 JSON；
    若 real_time_print 为 True，每解析到一条即在控制台打印。
    返回所有解析到的事件列表。
    """
    buffer = ""
    events = []
    for chunk in response.iter_content(chunk_size=4096):
        if chunk:
            buffer += chunk.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            event_block, buffer = buffer.split("\n\n", 1)
            event_block = event_block.strip()
            if not event_block or event_block.startswith(":"):
                continue
            for line in event_block.split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        data = json.loads(payload)
                        events.append(data)
                        if real_time_print:
                            print("[STREAM]", json.dumps(data, ensure_ascii=False))
                    except json.JSONDecodeError:
                        pass
                    break
    if buffer.strip():
        for line in buffer.split("\n"):
            if line.strip().startswith("data:"):
                payload = line.strip()[5:].strip()
                try:
                    data = json.loads(payload)
                    events.append(data)
                    if real_time_print:
                        print("[STREAM]", json.dumps(data, ensure_ascii=False))
                except json.JSONDecodeError:
                    pass
                break
    return events


def chat_stream_post(base_url, query, session_id=None, real_time_print=True):
    """发送流式请求，返回 (events, success)。"""
    url = f"{base_url.rstrip('/')}/chat/stream"
    payload = {
        "query": query,
        "tenant_id": 0,
        "user_id": None,
        "top_k": 5,
    }
    if session_id is not None:
        payload["session_id"] = session_id
    try:
        r = requests.post(
            url,
            json=payload,
            stream=True,
            timeout=60,
            headers={"Accept": "text/event-stream"},
        )
        r.raise_for_status()
        events = parse_sse_stream(r, real_time_print=real_time_print)
        return events, True
    except requests.RequestException as e:
        print(f"请求失败: {e}", file=sys.stderr)
        return [], False


def run_test_multiturn(base_url):
    """多轮对话：第一轮不完整 -> 引导年份/部门；第二轮同 session 补全 -> 触发 RAG。"""
    print("\n========== 多轮对话测试 ==========")
    session_id = str(uuid.uuid4())
    print(f"使用 session_id: {session_id}")

    # 第一轮：不完整指令
    print("\n--- 第一轮：查询财务报表（不提供年份/部门）---")
    events1, ok1 = chat_stream_post(
        base_url, "查询财务报表", session_id=session_id, real_time_print=True
    )
    assert ok1, "第一轮请求应成功"
    slot_asking_events = [e for e in events1 if e.get("type") == "slot_asking"]
    assert slot_asking_events, "第一轮应返回 type=slot_asking 的补槽引导"
    message = slot_asking_events[0].get("message", "")
    assert "年份" in message or "部门" in message, (
        f"第一轮应引导用户提供「年份」或「部门」，当前 message: {message}"
    )
    print("[PASS] 第一轮：返回内容在引导用户提供「年份」或「部门」")

    # 第二轮：同一 session 补全信息
    print("\n--- 第二轮：同一 session 发送「2025年，技术部」---")
    events2, ok2 = chat_stream_post(
        base_url, "2025年，技术部", session_id=session_id, real_time_print=True
    )
    assert ok2, "第二轮请求应成功"
    slot_asking_2 = [e for e in events2 if e.get("type") == "slot_asking"]
    citations_2 = [e for e in events2 if "citations" in e and e.get("citations") is not None]
    answers_2 = [e for e in events2 if e.get("answer")]
    assert (
        citations_2 or answers_2
    ), "第二轮应触发 RAG 检索（出现 citations 或 answer），当前未发现"
    print("[PASS] 第二轮：已成功触发 RAG 检索逻辑")


def run_test_isolation(base_url):
    """隔离性：不同 session_id 槽位互不干扰。"""
    print("\n========== 隔离性测试 ==========")
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())
    print(f"Session A: {session_a}")
    print(f"Session B: {session_b}")

    # A：不完整指令，应收到补槽
    print("\n--- Session A：查询财务报表 ---")
    events_a, ok_a = chat_stream_post(
        base_url, "查询财务报表", session_id=session_a, real_time_print=True
    )
    assert ok_a, "Session A 请求应成功"
    slot_a = [e for e in events_a if e.get("type") == "slot_asking"]
    assert slot_a, "Session A 应收到 slot_asking"

    # B：不完整指令，也应收到补槽（独立于 A）
    print("\n--- Session B：查询财务报表 ---")
    events_b, ok_b = chat_stream_post(
        base_url, "查询财务报表", session_id=session_b, real_time_print=True
    )
    assert ok_b, "Session B 请求应成功"
    slot_b = [e for e in events_b if e.get("type") == "slot_asking"]
    assert slot_b, "Session B 应收到 slot_asking（与 A 互不干扰）"

    # A：补全，应走 RAG
    print("\n--- Session A 补全：2025年，技术部 ---")
    events_a2, _ = chat_stream_post(
        base_url, "2025年，技术部", session_id=session_a, real_time_print=True
    )
    assert [e for e in events_a2 if "citations" in e or e.get("answer")], (
        "Session A 补全后应触发 RAG"
    )

    # B：未补全，再发一条不完整，仍应收到补槽（B 的槽位仍是空的）
    print("\n--- Session B 再发不完整：只看总结 ---")
    events_b2, _ = chat_stream_post(
        base_url, "只看总结", session_id=session_b, real_time_print=True
    )
    slot_b2 = [e for e in events_b2 if e.get("type") == "slot_asking"]
    assert slot_b2, "Session B 未补全时再问仍应收到 slot_asking（槽位隔离）"
    print("[PASS] 隔离性：不同 session 槽位互不干扰")


def main():
    parser = argparse.ArgumentParser(description="槽位填充流式接口自动化测试")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API 根地址，默认 {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--skip-multiturn",
        action="store_true",
        help="跳过多轮对话测试",
    )
    parser.add_argument(
        "--skip-isolation",
        action="store_true",
        help="跳过隔离性测试",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    print(f"Base URL: {base_url}")
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        print("服务健康检查通过")
    except requests.RequestException as e:
        print(f"服务不可用: {e}，请先启动 API（如 uvicorn app.api.main:app）", file=sys.stderr)
        sys.exit(1)

    failed = []
    if not args.skip_multiturn:
        try:
            run_test_multiturn(base_url)
        except AssertionError as e:
            failed.append(("多轮对话", str(e)))
    if not args.skip_isolation:
        try:
            run_test_isolation(base_url)
        except AssertionError as e:
            failed.append(("隔离性", str(e)))

    if failed:
        print("\n========== 失败 ==========")
        for name, msg in failed:
            print(f"  {name}: {msg}")
        sys.exit(1)
    print("\n========== 全部通过 ==========")


if __name__ == "__main__":
    main()
