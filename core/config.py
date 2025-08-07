import os
from dataclasses import dataclass
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class AppConfig:
    # Environment
    env: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 5000

    # API limits
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 100

    # Cache
    cache_type: str = "simple"
    cache_timeout: int = 3600
    cache_threshold: int = 100

    # Data
    cache_dir: str = "cache"

    # Model
    model_dir: str = "models"
    default_model: str = "cnn_bilstm_attention"

    # Risk
    max_position_risk: float = 0.01
    max_portfolio_risk: float = 0.02

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            env=Environment(os.getenv("ENV", "development")),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", 5000)),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_MINUTE", 10)),
            rate_limit_per_hour=int(os.getenv("RATE_LIMIT_HOUR", 100)),
        )


# Global config instance
config = AppConfig.from_env()
