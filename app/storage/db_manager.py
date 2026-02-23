"""
数据库连接池管理器
使用 SQLAlchemy 2.0 异步引擎和连接池
"""
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase
from app.config.config import settings


class Base(DeclarativeBase):
    """ORM 基类"""
    pass


class DatabaseManager:
    """数据库连接池单例管理器"""
    
    _instance = None
    _engine: Optional[AsyncEngine] = None
    _session_factory: Optional[async_sessionmaker[AsyncSession]] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._engine is None:
            self._initialize()
    
    def _initialize(self):
        """初始化数据库引擎和会话工厂"""
        # 创建异步引擎
        self._engine = create_async_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout,
            pool_recycle=settings.db_pool_recycle,
            echo=False,  # 生产环境设为 False，开发环境可设为 True
            future=True,
        )
        
        # 创建异步会话工厂
        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    
    @property
    def engine(self) -> AsyncEngine:
        """获取数据库引擎"""
        if self._engine is None:
            self._initialize()
        return self._engine
    
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """获取会话工厂"""
        if self._session_factory is None:
            self._initialize()
        return self._session_factory
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        获取数据库会话的上下文管理器
        
        使用示例:
            async with db_manager.get_session() as session:
                result = await session.execute(select(User))
                users = result.scalars().all()
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self):
        """关闭数据库连接池"""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
    
    async def create_tables(self):
        """创建所有表（仅用于开发/测试）"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def drop_tables(self):
        """删除所有表（仅用于开发/测试）"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)


# 全局数据库管理器实例
db_manager = DatabaseManager()
