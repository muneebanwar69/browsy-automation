"""
Configuration management for Browsy.

Supports multiple configuration sources:
- Environment variables
- Config files (YAML/JSON)
- Constructor arguments
- Default values
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BrowsyConfig(BaseSettings):
    """
    Configuration for BrowsyEngine.
    
    Configuration precedence (highest to lowest):
    1. Constructor arguments
    2. Environment variables (prefixed with BROWSY_)
    3. Config file (~/.browsy/config.yaml or custom path)
    4. Default values
    
    Examples:
        # Using constructor
        >>> config = BrowsyConfig(openai_api_key="sk-...")
        
        # Using environment variable
        >>> os.environ["BROWSY_OPENAI_API_KEY"] = "sk-..."
        >>> config = BrowsyConfig()
        
        # Using config file
        >>> config = BrowsyConfig.from_file("~/.browsy/config.yaml")
    """
    
    model_config = SettingsConfigDict(
        env_prefix="BROWSY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ============ OpenAI Configuration ============
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key (required)",
    )
    
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use",
    )
    
    max_tokens: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum tokens for LLM responses",
    )
    
    # ============ MCP Configuration ============
    mcp_config_path: Optional[str] = Field(
        default=None,
        description="Path to MCP config YAML file",
    )
    
    mcp_app_name: str = Field(
        default="browsy_automation",
        description="Name for MCP application",
    )
    
    playwright_headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )
    
    playwright_command: str = Field(
        default="npx",
        description="Command to run Playwright MCP server",
    )
    
    playwright_args: List[str] = Field(
        default_factory=lambda: ["@playwright/mcp@latest", "--headless"],
        description="Arguments for Playwright MCP server",
    )
    
    # ============ Session Configuration ============
    use_history: bool = Field(
        default=True,
        description="Use conversation history by default",
    )
    
    session_timeout: int = Field(
        default=3600,
        ge=60,
        description="Session timeout in seconds",
    )
    
    # ============ Logging Configuration ============
    log_level: str = Field(
        default="warning",
        description="Logging level (debug, info, warning, error)",
    )
    
    log_path: Optional[str] = Field(
        default=None,
        description="Path for log files",
    )
    
    # ============ API Server Configuration (Optional) ============
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )
    
    api_port: int = Field(
        default=5000,
        ge=1024,
        le=65535,
        description="API server port",
    )
    
    api_cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS allowed origins",
    )
    
    # ============ Methods ============
    
    @classmethod
    def from_file(cls, path: str) -> "BrowsyConfig":
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            path: Path to config file
            
        Returns:
            BrowsyConfig instance
            
        Example:
            >>> config = BrowsyConfig.from_file("~/.browsy/config.yaml")
        """
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif path.suffix == ".json":
                import json
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls(**data)
    
    def to_file(self, path: str):
        """
        Save configuration to a YAML file.
        
        Args:
            path: Path to save config file
            
        Example:
            >>> config.to_file("~/.browsy/config.yaml")
        """
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            # Exclude empty values and sensitive data
            data = self.model_dump(exclude_none=True, exclude={"openai_api_key"})
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_mcp_config(self) -> Dict[str, Any]:
        """
        Generate MCP configuration dict.
        
        Returns:
            MCP config dictionary for mcp-agent
        """
        args = self.playwright_args.copy()
        if self.playwright_headless and "--headless" not in args:
            args.append("--headless")
        
        return {
            "execution_engine": "asyncio",
            "logger": {
                "transports": ["file"] if self.log_path else [],
                "level": self.log_level,
                "progress_display": False,
                "path_settings": {
                    "path_pattern": self.log_path or "logs/browsy-{unique_id}.jsonl",
                    "unique_id": "timestamp",
                    "timestamp_format": "%Y%m%d_%H%M%S",
                },
            },
            "mcp": {
                "servers": {
                    "playwright": {
                        "command": self.playwright_command,
                        "args": args,
                    }
                }
            },
            "openai": {
                "default_model": self.openai_model,
            },
        }
    
    def validate_api_key(self) -> bool:
        """
        Validate that OpenAI API key is set.
        
        Returns:
            True if API key is valid (non-empty and starts with sk-)
        """
        if not self.openai_api_key:
            return False
        return self.openai_api_key.startswith("sk-")
    
    def __repr__(self) -> str:
        """Safe repr that hides API key."""
        masked_key = f"{self.openai_api_key[:7]}...{self.openai_api_key[-4:]}" if self.openai_api_key else "NOT_SET"
        return (
            f"BrowsyConfig(openai_api_key='{masked_key}', "
            f"openai_model='{self.openai_model}', "
            f"playwright_headless={self.playwright_headless})"
        )
