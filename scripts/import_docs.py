#!/usr/bin/env python3
"""
独立文档导入脚本：扫描指定目录下的 PDF，调用 DocumentIngestionPipeline.ingest_pdf 按需入库。
与问答服务解耦，仅在需要时运行以构建/更新索引。

用法:
  python scripts/import_docs.py [目录路径]
  默认目录: ./data/raw

示例:
  python scripts/import_docs.py
  python scripts/import_docs.py /path/to/pdfs
"""
import asyncio
import argparse
import logging
import sys
from pathlib import Path

# 将项目根目录加入 path，便于以脚本方式运行
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="扫描目录中的 PDF 并调用摄取流水线入库（幂等，已解析的文档会跳过）"
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default="./data/raw",
        help="要扫描的目录路径，默认: ./data/raw",
    )
    parser.add_argument(
        "--tenant-id",
        type=int,
        default=0,
        help="租户 ID，默认 0",
    )
    parser.add_argument(
        "--doc-type",
        default="annual_report",
        help="文档类型，默认 annual_report",
    )
    parser.add_argument(
        "--source",
        default="import_script",
        help="来源标识，默认 import_script",
    )
    args = parser.parse_args()

    folder = Path(args.directory).resolve()
    if not folder.exists():
        logger.error("目录不存在: %s", folder)
        sys.exit(1)
    if not folder.is_dir():
        logger.error("路径不是目录: %s", folder)
        sys.exit(1)

    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))
    pdf_files = sorted(set(pdf_files))
    if not pdf_files:
        logger.info("目录下未找到 PDF 文件: %s", folder)
        return

    logger.info("开始导入，目录=%s, 共 %d 个 PDF", folder, len(pdf_files))

    async def run():
        from app.pipeline.ingestion import DocumentIngestionPipeline

        pipeline = DocumentIngestionPipeline()
        processed = 0
        skipped = 0
        failed = 0
        for pdf_path in pdf_files:
            try:
                result = await pipeline.ingest_pdf(
                    file_path=str(pdf_path),
                    tenant_id=args.tenant_id,
                    doc_type=args.doc_type,
                    source=args.source,
                )
                if result.get("success"):
                    if result.get("message", "").find("跳过") != -1:
                        skipped += 1
                        logger.info("跳过（已存在）: %s", pdf_path.name)
                    else:
                        processed += 1
                        logger.info(
                            "入库成功: %s -> document_id=%s, chunks=%s",
                            pdf_path.name,
                            result.get("document_id"),
                            result.get("chunks_count", 0),
                        )
                else:
                    failed += 1
                    logger.warning("入库失败: %s -> %s", pdf_path.name, result.get("error"))
            except Exception as e:
                failed += 1
                logger.exception("处理异常 %s: %s", pdf_path.name, e)

        logger.info("导入结束: 成功=%d, 跳过=%d, 失败=%d", processed, skipped, failed)

    asyncio.run(run())


if __name__ == "__main__":
    main()
