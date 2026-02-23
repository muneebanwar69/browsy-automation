"""
Standalone Browsy API server.

This creates a standalone FastAPI server for Browsy automation.
Can be used as a microservice that other applications can call.
"""

import os
from browsy.api import create_app
from browsy import BrowsyConfig


def main():
    """Run standalone Browsy API server."""
    
    # Load config from environment or file
    config_file = os.getenv("BROWSY_CONFIG_FILE")
    
    if config_file:
        print(f"Loading config from: {config_file}")
        config = BrowsyConfig.from_file(config_file)
    else:
        print("Using environment variables for configuration")
        config = BrowsyConfig()  # Loads from environment
    
    # Validate API key
    if not config.validate_api_key():
        print("❌ Error: OpenAI API key not configured!")
        print("Set via:")
        print("  - BROWSY_OPENAI_API_KEY environment variable")
        print("  - BROWSY_CONFIG_FILE environment variable pointing to config file")
        return
    
    # Create app
    app = create_app(config=config, include_cors=True)
    
    # Run server
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    print(f"\n🚀 Starting Browsy API Server")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Docs: http://{host}:{port}/docs")
    print(f"   Health: http://{host}:{port}/api/health\n")
    
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
