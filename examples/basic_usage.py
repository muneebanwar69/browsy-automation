"""
Basic usage example for Browsy Automation.

This example shows how to use BrowsyEngine to execute simple web automation tasks.
"""

import asyncio
from browsy import BrowsyEngine, BrowsyConfig


async def main():
    """Run basic automation examples."""
    
    # Create config (or use environment variable BROWSY_OPENAI_API_KEY)
    config = BrowsyConfig(
        openai_api_key="YOUR_API_KEY_HERE",  # Replace with your API key
        playwright_headless=True,
    )
    
    # Example 1: Using context manager (recommended)
    print("=" * 60)
    print("Example 1: Simple query with context manager")
    print("=" * 60)
    
    async with BrowsyEngine(config=config) as engine:
        result = await engine.execute_sync("Go to example.com and get the page title")
        if result.success:
            print(f"\nResult:\n{result.result}")
            print(f"\nElapsed: {result.elapsed}s")
        else:
            print(f"\nError: {result.error}")
    
    # Example 2: Progress streaming
    print("\n" + "=" * 60)
    print("Example 2: Query with progress streaming")
    print("=" * 60 + "\n")
    
    engine = BrowsyEngine(config=config)
    await engine.initialize()
    
    async for event in engine.execute("Navigate to github.com and find the trending Python repos"):
        if event.type == "progress":
            print(f"[{event.progress:3d}%] {event.stage:15s} - {event.message}")
        elif event.type == "result":
            print(f"\n✅ Success!\n")
            print(event.result)
        elif event.type == "error":
            print(f"\n❌ Error: {event.message}")
    
    await engine.cleanup()
    
    # Example 3: Multi-step conversation
    print("\n" + "=" * 60)
    print("Example 3: Multi-step conversation with session")
    print("=" * 60 + "\n")
    
    async with BrowsyEngine(config=config) as engine:
        # Step 1: Navigate to a site
        result1 = await engine.execute_sync("Go to python.org")
        print(f"Step 1 Result:\n{result1.result[:200]}...\n")
        
        # Step 2: Use same session for context
        result2 = await engine.execute_sync(
            "Find the latest Python version mentioned",
            session_id=result1.session_id
        )
        print(f"Step 2 Result:\n{result2.result}\n")
        
        # Get session stats
        session = engine.get_session(result1.session_id)
        print(f"Session queries: {session.query_count}")
    
    # Example 4: Statistics
    print("\n" + "=" * 60)
    print("Example 4: Usage statistics")
    print("=" * 60 + "\n")
    
    async with BrowsyEngine(config=config) as engine:
        # Run a few queries
        await engine.execute_sync("Go to example.com")
        await engine.execute_sync("Go to github.com")
        await engine.execute_sync("Go to stackoverflow.com")
        
        # Get stats
        stats = engine.get_stats()
        print(f"Total queries: {stats['total_queries']}")
        print(f"Successful: {stats['successful_queries']}")
        print(f"Success rate: {stats['success_rate']}%")
        print(f"Avg response time: {stats['avg_response_time']}s")
        print(f"Active sessions: {stats['active_sessions']}")


if __name__ == "__main__":
    asyncio.run(main())
