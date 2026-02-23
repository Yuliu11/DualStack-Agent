"""
数据库清理脚本
彻底 TRUNCATE 掉 chunks 和 documents 表，确保环境绝对干净
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sqlalchemy import text
from app.storage.db_manager import db_manager


async def cleanup_database():
    """清理数据库表"""
    print("=" * 60)
    print("开始清理数据库...")
    print("=" * 60)
    
    try:
        async with db_manager.get_session() as session:
            # 获取表名列表（按依赖顺序）
            tables_to_truncate = [
                "chunks",
                "doc_versions", 
                "tables",
                "embeddings",
                "documents"
            ]
            
            print("\n清理顺序（考虑外键依赖）:")
            for i, table in enumerate(tables_to_truncate, 1):
                print(f"  {i}. {table}")
            
            print("\n开始执行 TRUNCATE...")
            
            # 先禁用外键检查（MySQL）
            await session.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
            print("  [OK] 已禁用外键检查")
            
            # 逐个 TRUNCATE 表
            for table in tables_to_truncate:
                try:
                    await session.execute(text(f"TRUNCATE TABLE `{table}`"))
                    await session.commit()
                    print(f"  [OK] 已清理表: {table}")
                except Exception as e:
                    print(f"  [ERROR] 清理表 {table} 失败: {e}")
                    # 继续清理其他表
            
            # 重新启用外键检查
            await session.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
            print("  [OK] 已重新启用外键检查")
            
            print("\n" + "=" * 60)
            print("数据库清理完成！")
            print("=" * 60)
            
    except Exception as e:
        print(f"\n发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(cleanup_database())
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
