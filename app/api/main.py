"""
FastAPI 主入口
提供文档摄取和问答接口
"""
import os
import logging
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.pipeline import document_pipeline
from app.service.rag_engine import rag_service
from app.config.config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    import asyncio
    
    # 初始化向量库
    try:
        from app.retrieval.engine import RetrievalEngine
        retrieval_engine = RetrievalEngine()
        # 触发向量库初始化（如果使用 FAISS，会尝试加载索引）
        if retrieval_engine.vector_store.vector_store_type == "faiss":
            index_path = retrieval_engine.vector_store.index_path
            index_file = Path(index_path)
            if not index_file.exists():
                logger.warning(
                    f"FAISS 索引文件不存在: {index_file.absolute()}. "
                    f"请运行 ingest_local.py 或上传文档以创建索引。"
                )
            else:
                logger.info(f"向量库初始化完成: {index_file.absolute()}")
    except Exception as e:
        logger.error(f"向量库初始化失败: {e}", exc_info=True)
    
    # 获取 data/raw 文件夹路径
    raw_folder = Path("./data/raw")
    
    if raw_folder.exists() and raw_folder.is_dir():
        logger.info(f"启动时自动扫描文件夹: {raw_folder.absolute()}")
        
        # 在后台任务中执行扫描，避免阻塞启动
        async def scan_folder():
            try:
                result = await document_pipeline.process_local_folder(
                    folder_path=str(raw_folder.absolute()),
                    tenant_id=0,
                    doc_type="annual_report",
                    source="auto_scan"
                )
                
                if result.get('success'):
                    logger.info(
                        f"自动扫描完成: 处理 {result.get('processed', 0)} 个文件, "
                        f"跳过 {result.get('skipped', 0)} 个文件, "
                        f"失败 {result.get('failed', 0)} 个文件"
                    )
                else:
                    logger.warning(f"自动扫描失败: {result.get('error')}")
            except Exception as e:
                logger.error(f"自动扫描时发生错误: {e}", exc_info=True)
        
        # 启动后台任务
        asyncio.create_task(scan_folder())
    else:
        logger.info(f"data/raw 文件夹不存在，跳过自动扫描: {raw_folder.absolute()}")
    
    yield  # 应用运行中
    
    # 关闭时执行（如果需要）
    pass


# 创建 FastAPI 应用
app = FastAPI(
    title="RAG Agent API",
    description="文档问答系统 API",
    version="0.1.0",
    lifespan=lifespan
)

# 配置 CORS
# 注意：当 allow_origins=["*"] 时，不能设置 allow_credentials=True
# 如果需要 credentials，需要明确指定允许的域名
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（开发环境）
    # allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "file://"],  # 生产环境应限制具体域名
    allow_credentials=False,  # 当 allow_origins=["*"] 时，必须为 False
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # 允许的HTTP方法
    allow_headers=["*"],  # 允许所有请求头
    expose_headers=["*"],  # 暴露所有响应头
)


# 请求/响应模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str = Field(..., description="用户查询", min_length=1, max_length=1000)
    tenant_id: int = Field(default=0, description="租户ID")
    user_id: Optional[int] = Field(default=None, description="用户ID")
    top_k: Optional[int] = Field(default=5, description="检索返回的 Top-K 数量", ge=1, le=20)


class ChatResponse(BaseModel):
    """聊天响应模型"""
    success: bool
    request_id: Optional[int] = None
    answer: str
    confidence: float
    is_refused: bool
    model_name: Optional[str] = None
    citations: list = []
    retrieved_chunks_count: int = 0
    cited_chunks_count: int = 0
    latency_ms: int = 0
    error: Optional[str] = None


class IngestResponse(BaseModel):
    """文档摄取响应模型"""
    success: bool
    document_id: Optional[int] = None
    version_id: Optional[int] = None
    version_no: Optional[int] = None
    chunks_count: int = 0
    reused_chunks_count: int = 0
    new_chunks_count: int = 0
    reusable_embeddings_count: int = 0
    tables_count: int = 0
    file_sha256: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "服务器内部错误，请稍后重试",
            "detail": str(exc) if settings.db_host == "localhost" else None  # 开发环境显示详细错误
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """HTTP 异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail
        }
    )


# 健康检查
@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "RAG Agent"}


# 文档摄取接口
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF 文件"),
    tenant_id: int = 0,
    doc_type: str = "annual_report",
    source: str = "upload",
    title: Optional[str] = None,
    company_name: Optional[str] = None,
    stock_code: Optional[str] = None,
    year: Optional[int] = None
):
    """
    上传 PDF 文件，触发 Pipeline 进行解析和向量入库
    
    Args:
        file: PDF 文件
        tenant_id: 租户ID
        doc_type: 文档类型
        source: 来源
        title: 文档标题
        company_name: 公司名称
        stock_code: 股票代码
        year: 年份
    
    Returns:
        摄取结果
    """
    # 验证文件类型
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    
    # 保存上传的文件到临时目录
    upload_dir = Path("./uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        # 保存文件
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"开始处理文档: {file.filename}")
        
        # 调用 Pipeline 处理
        result = await document_pipeline.ingest_pdf(
            file_path=str(file_path),
            tenant_id=tenant_id,
            doc_type=doc_type,
            source=source,
            title=title,
            company_name=company_name,
            stock_code=stock_code,
            year=year
        )
        
        # 清理临时文件
        try:
            os.remove(file_path)
        except:
            pass
        
        if result.get('success'):
            logger.info(
                f"文档处理成功: document_id={result.get('document_id')}, "
                f"chunks={result.get('chunks_count')}"
            )
        else:
            logger.error(f"文档处理失败: {result.get('error')}")
        
        return IngestResponse(**result)
    
    except Exception as e:
        # 清理临时文件
        try:
            if file_path.exists():
                os.remove(file_path)
        except:
            pass
        
        logger.error(f"文档摄取失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败: {str(e)}"
        )


# 问答接口
@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest
):
    """
    接收用户提问，返回带引用来源的答案
    
    Args:
        request: 聊天请求
    
    Returns:
        聊天响应
    """
    try:
        logger.info(f"收到查询: {request.query[:50]}...")
        
        # 调用 RAG 服务
        result = await rag_service.query(
            query=request.query,
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        # 确保 result 包含所有必需字段
        if not isinstance(result, dict):
            result = {}
        
        # 验证并补全必需字段
        if result.get('success'):
            logger.info(
                f"查询成功: request_id={result.get('request_id')}, "
                f"confidence={result.get('confidence', 0.0):.2f}"
            )
        else:
            logger.warning(f"查询失败: {result.get('error', '未知错误')}")
        
        # 构建完整的响应，确保所有字段都有默认值
        return ChatResponse(
            success=result.get('success', False),
            request_id=result.get('request_id'),
            answer=result.get('answer', ''),
            confidence=result.get('confidence', 0.0),
            is_refused=result.get('is_refused', False),
            model_name=result.get('model_name'),
            citations=result.get('citations', []),
            retrieved_chunks_count=result.get('retrieved_chunks_count', 0),
            cited_chunks_count=result.get('cited_chunks_count', 0),
            latency_ms=result.get('latency_ms', 0),
            error=result.get('error')
        )
    
    except TypeError as e:
        # Pydantic 验证错误或其他类型错误
        logger.error(f"类型错误或验证失败: {e}", exc_info=True)
        return ChatResponse(
            success=False,
            request_id=None,
            answer="抱歉，系统内部错误",
            confidence=0.0,
            is_refused=True,
            model_name=None,
            citations=[],
            retrieved_chunks_count=0,
            cited_chunks_count=0,
            latency_ms=0,
            error="系统内部错误，请稍后重试"
        )
    except Exception as e:
        logger.error(f"查询处理失败: {e}", exc_info=True)
        
        # 根据异常类型返回不同的错误信息
        error_msg = "处理失败，请稍后重试"
        if "circuit" in str(e).lower() or "熔断" in str(e):
            error_msg = "服务暂时不可用，请稍后重试"
        elif "connection" in str(e).lower() or "连接" in str(e):
            error_msg = "数据库连接失败，请稍后重试"
        elif "timeout" in str(e).lower():
            error_msg = "请求超时，请稍后重试"
        elif "validation" in str(e).lower() or "pydantic" in str(e).lower():
            error_msg = "请求参数验证失败，请检查输入"
        else:
            # 开发环境显示详细错误，生产环境显示通用错误
            if settings.db_host == "localhost":
                error_msg = f"处理失败: {str(e)}"
            else:
                error_msg = "处理失败，请稍后重试"
        
        # 返回完整的 ChatResponse 结构，确保所有必需字段都有值
        # 强制符合 Pydantic 模型要求，避免前端崩溃
        return ChatResponse(
            success=False,
            request_id=None,
            answer="抱歉，系统内部错误",
            confidence=0.0,
            is_refused=True,
            model_name=None,
            citations=[],
            retrieved_chunks_count=0,
            cited_chunks_count=0,
            latency_ms=0,
            error=error_msg
        )


# 根路径
@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "RAG Agent API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest",
            "chat": "/chat"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
