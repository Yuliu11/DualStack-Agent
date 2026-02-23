"""
RAG Agent 应用启动脚本
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式：自动重载
        log_level="info"
    )
