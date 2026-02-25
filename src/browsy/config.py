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
    
    # ============ LLM Configuration ============
    openai_api_key: str = Field(
        default="",
        description="LLM API key (OpenAI, LLM Gateway, or any OpenAI-compatible provider)",
    )
    
    openai_base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL (e.g., https://api.llmgateway.io/v1 for LLM Gateway)",
    )
    
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Model to use (e.g., gpt-4o-mini, gpt-4o)",
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
    
    # ============ Performance Configuration ============
    enable_browser_reuse: bool = Field(
        default=True,
        description="Reuse browser instances across requests for better performance",
    )
    
    enable_resource_blocking: bool = Field(
        default=True,
        description="Block unnecessary resources based on task type",
    )
    
    enable_session_caching: bool = Field(
        default=True,
        description="Cache login sessions and cookies",
    )
    
    cache_ttl: int = Field(
        default=3600,
        ge=60,
        description="Cache time-to-live in seconds",
    )
    
    max_browser_pool_size: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum number of browser instances in pool",
    )
    
    enable_performance_metrics: bool = Field(
        default=True,
        description="Track and report performance metrics",
    )
    
    resource_strategy: str = Field(
        default="auto",
        description="Resource loading strategy: auto, screenshot, data, interact, speed",
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
        
        # Handle headless flag properly
        if self.playwright_headless:
            # Add --headless if not present
            if "--headless" not in args:
                args.append("--headless")
        else:
            # Remove --headless if present (for headed mode)
            args = [arg for arg in args if arg != "--headless"]
        
        # Build OpenAI config with credentials
        openai_config = {
            "default_model": self.openai_model,
            "api_key": self.openai_api_key,
        }
        
        # Add base URL if specified (for OpenRouter, LLM Gateway, etc.)
        if self.openai_base_url:
            openai_config["base_url"] = self.openai_base_url
        
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
            "openai": openai_config,
        }
    
    def validate_api_key(self) -> bool:
        """
        Validate that API key is set.
        
        Returns:
            True if API key is valid (non-empty)
        """
        return bool(self.openai_api_key and self.openai_api_key.strip())
    
    def __repr__(self) -> str:
        """Safe repr that hides API key."""
        masked_key = f"{self.openai_api_key[:7]}...{self.openai_api_key[-4:]}" if self.openai_api_key else "NOT_SET"
        return (
            f"BrowsyConfig(openai_api_key='{masked_key}', "
            f"openai_model='{self.openai_model}', "
            f"playwright_headless={self.playwright_headless})"
        )
