import asyncio
# 假设你的代码在 app/router.py。如果不是，请根据 dir 的结果修改下一行
from app.router import intent_router 

async def test():
    test_queries = [
        "常磊科技的CEO是谁？", # 预期: FACT
        "对比一下大树科技和常磊科技的去年净利润差多少", # 预期: CALC
        "请帮我总结一下这份文档的核心内容", # 预期: SUMMARY
        "增长率是怎么算的？" # 预期: CALC 或 FACT
    ]
    
    print("\n" + "="*30)
    print("🚀 Agent 意图识别压力测试")
    print("="*30 + "\n")

    for q in test_queries:
        try:
            # 获取意图
            res = await intent_router.get_intent(q)
            print(f"🔍 问题: {q}")
            print(f"🎯 识别: {res}\n")
        except Exception as e:
            print(f"❌ 出错了: {e}")

if __name__ == "__main__":
    asyncio.run(test())