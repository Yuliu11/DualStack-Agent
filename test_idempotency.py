import asyncio
import httpx

async def send_request():
    url = "http://127.0.0.1:8000/chat" 
    data = {
        "user_id": 12345, # 注意：你的模型里 user_id 是 int
        "query": "对比常磊科技2024和2022毛利率",
        "tenant_id": 0
    }
    
    # 设置超时时间为 60 秒，因为 RAG 比较慢
    async with httpx.AsyncClient(timeout=60.0) as client:
        print("🚀 正在发送并发请求测试幂等性...")
        tasks = [
            client.post(url, json=data),
            client.post(url, json=data)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        for i, res in enumerate(responses):
            print(f"📡 请求 {i+1} 返回 - 状态码: {res.status_code}")
            print(f"📦 响应内容: {res.json()}\n")

if __name__ == "__main__":
    asyncio.run(send_request())