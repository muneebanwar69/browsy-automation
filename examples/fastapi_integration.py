"""
FastAPI integration example.

Shows how to embed Browsy into an existing FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from browsy.api import create_browsy_router
from browsy import BrowsyConfig


# Create your main app
app = FastAPI(
    title="My Application with Browsy",
    description="Example of embedding Browsy into an existing FastAPI app",
    version="1.0.0",
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your existing routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to my app!",
        "features": {
            "browsy": "/api/browsy/health",
            "docs": "/docs",
        }
    }


@app.get("/api/hello")
async def hello():
    """Example endpoint."""
    return {"message": "Hello from my app!"}


# Add Browsy router
config = BrowsyConfig(
    openai_api_key="YOUR_API_KEY_HERE",  # Or use environment variable
    playwright_headless=True,
)

browsy_router = create_browsy_router(config)
app.include_router(browsy_router, prefix="/api/browsy", tags=["Automation"])


# Run with: uvicorn fastapi_integration:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
