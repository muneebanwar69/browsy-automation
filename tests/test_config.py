"""
Tests for Browsy configuration.
"""

import pytest
from browsy.config import BrowsyConfig


def test_config_defaults():
    """Test default configuration values."""
    config = BrowsyConfig(openai_api_key="sk-test123")
    
    assert config.openai_model == "gpt-4o-mini"
    assert config.max_tokens == 10000
    assert config.playwright_headless is True
    assert config.use_history is True
    assert config.mcp_app_name == "browsy_automation"


def test_config_validation():
    """Test API key validation."""
    # Valid key
    config = BrowsyConfig(openai_api_key="sk-test123")
    assert config.validate_api_key() is True
    
    # Invalid key
    config_invalid = BrowsyConfig(openai_api_key="invalid")
    assert config_invalid.validate_api_key() is False
    
    # Empty key
    config_empty = BrowsyConfig(openai_api_key="")
    assert config_empty.validate_api_key() is False


def test_config_mcp_generation():
    """Test MCP config generation."""
    config = BrowsyConfig(
        openai_api_key="sk-test123",
        openai_model="gpt-4",
        playwright_headless=True,
    )
    
    mcp_config = config.get_mcp_config()
    
    assert "mcp" in mcp_config
    assert "servers" in mcp_config["mcp"]
    assert "playwright" in mcp_config["mcp"]["servers"]
    assert mcp_config["openai"]["default_model"] == "gpt-4"


def test_config_repr():
    """Test safe repr that masks API key."""
    config = BrowsyConfig(openai_api_key="sk-test123456789")
    repr_str = repr(config)
    
    assert "sk-test" in repr_str
    assert "123456789" not in repr_str  # Should be masked
    assert "..." in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
