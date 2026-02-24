import asyncio
import sys
import os

# 1. 动态添加项目根目录到 Python 路径
sys.path.append(os.getcwd())

# 2. 修正导入名称：从 rag_engine.py 导入 rag_service
try:
    from app.service.rag_engine import rag_service
    print("✅ 成功导入 rag_service")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

async def test_agent():
    # 测试一个典型的计算问题，触发你的 CALC 逻辑
    query = "对比一下常磊科技 2024 年和 2023 年的营收差值是多少？"
    
    print(f"\n🧐 正在发送请求: {query}")
    print("-" * 30)
    
    try:
        # 调用 RAG 主流程
        result = await rag_service.query(query)
        
        if result.get('success'):
            print("🤖 Agent 响应成功！")
            print(f"📝 最终回答: {result.get('answer')}")
            print(f"📊 使用模型: {result.get('model_name')}")
            print(f"⏱️ 耗时: {result.get('latency_ms')} ms")
        else:
            print(f"⚠️ 请求未成功: {result.get('answer')}")
            
    except Exception as e:
        print(f"💥 运行异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 解决 Windows 下可能出现的 ProactorEventLoop 警告
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(test_agent())