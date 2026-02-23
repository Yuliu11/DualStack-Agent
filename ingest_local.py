"""
本地文件夹批量扫描入库脚本
扫描 data/raw 文件夹下的所有 PDF 文件并处理
"""
import re
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.pipeline import document_pipeline


async def main():
    """主函数"""
    # 获取 data/raw 文件夹路径
    raw_folder = project_root / "data" / "raw"
    
    if not raw_folder.exists():
        print(f"错误: 文件夹不存在: {raw_folder.absolute()}")
        print(f"请创建文件夹: {raw_folder.absolute()}")
        return
    
    if not raw_folder.is_dir():
        print(f"错误: 路径不是文件夹: {raw_folder.absolute()}")
        return
    
    print(f"开始扫描文件夹: {raw_folder.absolute()}")
    print("-" * 60)
    
    # 调用批量处理函数
    result = await document_pipeline.process_local_folder(
        folder_path=str(raw_folder.absolute()),
        tenant_id=0,
        doc_type="annual_report",
        source="local_scan"
    )
    
    print("-" * 60)
    
    if result.get('success'):
        processed = result.get('processed', 0)
        updated = result.get('updated', 0)  # 覆盖更新的文件数
        skipped = result.get('skipped', 0)
        failed = result.get('failed', 0)
        total_files = result.get('total_files', 0)
        
        # 计算总 Chunk 数量
        total_chunks = 0
        for item in result.get('results', []):
            if item.get('status') == 'success':
                total_chunks += item.get('chunks_count', 0)
        
        # 打印结果（按照用户要求的格式）
        print(f"\n成功解析 {processed} 个文档，共切分为 {total_chunks} 个 Chunk")
        
        # 额外信息
        if updated > 0 or skipped > 0 or failed > 0:
            print(f"\n详细信息:")
            print(f"  总文件数: {total_files}")
            if updated > 0:
                print(f"  覆盖更新: {updated} 个文档")
            if skipped > 0:
                print(f"  跳过（已存在）: {skipped} 个文档")
            if failed > 0:
                print(f"  失败: {failed} 个文档")
        
        # 打印详细信息
        if result.get('results'):
            print("\n详细结果:")
            for item in result.get('results', []):
                file_name = item.get('file', 'unknown')
                status = item.get('status', 'unknown')
                
                if status == 'success':
                    doc_id = item.get('document_id', 'N/A')
                    chunks = item.get('chunks_count', 0)
                    is_update = item.get('is_update', False)
                    action = "覆盖更新" if is_update else "新增"
                    print(f"  [OK] {file_name}: {action} (文档ID: {doc_id}, Chunks: {chunks})")
                elif status == 'skipped':
                    doc_id = item.get('document_id', 'N/A')
                    print(f"  [SKIP] {file_name}: 跳过 (文档ID: {doc_id})")
                elif status == 'failed':
                    error = item.get('error', '未知错误')
                    print(f"  [ERROR] {file_name}: 失败 ({error})")
    else:
        print(f"扫描失败: {result.get('error', '未知错误')}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
