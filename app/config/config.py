"""
应用配置模块
使用 Pydantic-settings 管理配置，支持从 .env 文件和环境变量读取
"""
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path


class Settings(BaseSettings):
    """应用配置类"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # DeepSeek API 配置
    deepseek_api_key: str = Field(..., description="DeepSeek API 密钥")
    deepseek_base_url: Optional[str] = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API 基础 URL"
    )
    
    # 阿里云百炼 API 配置
    dashscope_api_key: str = Field(..., description="阿里云百炼 API 密钥")
    dashscope_base_url: Optional[str] = Field(
        default="https://dashscope.aliyuncs.com",
        description="阿里云百炼 API 基础 URL"
    )
    
    # 数据库配置
    db_host: str = Field(default="localhost", description="MySQL 主机地址")
    db_port: int = Field(default=3306, description="MySQL 端口")
    db_user: str = Field(..., description="MySQL 用户名")
    db_password: str = Field(..., description="MySQL 密码")
    db_name: str = Field(..., description="数据库名称")
    db_charset: str = Field(default="utf8mb4", description="数据库字符集")
    
    # Redis 配置
    redis_host: str = Field(default="localhost", description="Redis 主机地址")
    redis_port: int = Field(default=6379, description="Redis 端口")
    redis_password: Optional[str] = Field(default=None, description="Redis 密码")
    redis_db: int = Field(default=0, description="Redis 数据库编号")
    redis_decode_responses: bool = Field(default=True, description="Redis 响应解码")
    
    # 数据库连接池配置
    db_pool_size: int = Field(default=10, description="数据库连接池大小")
    db_max_overflow: int = Field(default=20, description="数据库连接池最大溢出")
    db_pool_timeout: int = Field(default=30, description="数据库连接池超时时间（秒）")
    db_pool_recycle: int = Field(default=3600, description="数据库连接回收时间（秒）")
    
    # Redis 连接池配置
    redis_max_connections: int = Field(default=50, description="Redis 最大连接数")
    
    @property
    def database_url(self) -> str:
        """构建数据库连接 URL"""
        return (
            f"mysql+aiomysql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}?charset={self.db_charset}"
        )
    
    @property
    def redis_url(self) -> str:
        """构建 Redis 连接 URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class YAMLConfig:
    """YAML 配置文件管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        self.config_path = Path(config_path)
        self._config: Optional[dict] = None
    
    def load(self) -> dict:
        """加载 YAML 配置"""
        if self._config is None:
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
        return self._config
    
    def get(self, key: str, default=None):
        """获取配置项（支持点号分隔的嵌套键）"""
        config = self.load()
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def reload(self):
        """重新加载配置"""
        self._config = None
        return self.load()


class PromptConfig:
    """Prompt 配置文件管理器"""
    
    def __init__(self, prompts_path: Optional[str] = None):
        if prompts_path is None:
            prompts_path = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
        self.prompts_path = Path(prompts_path)
        self._prompts: Optional[dict] = None
    
    def load(self) -> dict:
        """加载 Prompt 配置"""
        if self._prompts is None:
            if not self.prompts_path.exists():
                raise FileNotFoundError(f"Prompt配置文件不存在: {self.prompts_path}")
            with open(self.prompts_path, "r", encoding="utf-8") as f:
                self._prompts = yaml.safe_load(f) or {}
        return self._prompts
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        获取Prompt模板（支持点号分隔的嵌套键）
        
        Args:
            key: Prompt键，例如 "rag_answer.system_template"
            default: 默认值
        
        Returns:
            Prompt模板字符串，如果不存在则返回default
        """
        prompts = self.load()
        keys = key.split(".")
        value = prompts
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def format(self, key: str, **kwargs) -> str:
        """
        获取并格式化Prompt模板
        
        Args:
            key: Prompt键，例如 "rag_answer.system_template"
            **kwargs: 格式化参数
        
        Returns:
            格式化后的Prompt字符串
        """
        template = self.get(key)
        if template is None:
            raise ValueError(f"Prompt模板不存在: {key}")
        return template.format(**kwargs)
    
    def reload(self):
        """重新加载Prompt配置"""
        self._prompts = None
        return self.load()


# 全局配置实例
settings = Settings()
yaml_config = YAMLConfig()
prompt_config = PromptConfig()