"""
Example: Using Browsy with Performance Optimizations

This example demonstrates the performance optimizations in action:
- Browser reuse across requests
- Task-aware resource blocking
- Session caching
- Performance metrics
"""

import asyncio
from browsy import BrowsyEngine, BrowsyConfig


async def main():
    """Demonstrate performance optimizations."""
    
    # Configure Browsy with all optimizations enabled (default)
    config = BrowsyConfig(
        openai_api_key="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",
        openai_base_url="https://openrouter.ai/api/v1",
        openai_model="openai/gpt-4o-mini",
        playwright_headless=True,
        
        # Performance optimizations (all enabled by default)
        enable_browser_reuse=True,          # Reuse browser instances
        enable_resource_blocking=True,      # Smart resource blocking per task
        enable_session_caching=True,        # Cache login sessions
        enable_performance_metrics=True,    # Track performance
        
        cache_ttl=3600,                     # Cache for 1 hour
        max_browser_pool_size=1,            # Single browser instance
    )
    
    print("=" * 80)
    print("🚀 Browsy Performance Optimization Demo")
    print("=" * 80)
    print()
    
    async with BrowsyEngine(config) as engine:
        
        # ========== Test 1: Fast Interaction (Login) ==========
        print("📝 Test 1: Fast Interaction Mode (Login)")
        print("-" * 80)
        
        result1 = await engine.execute_sync(
            "Go to example.com and click any button you find"
        )
        
        if result1.success:
            print(f"✅ Success in {result1.elapsed}s (INTERACTION mode)")
            print(f"   Result: {result1.result[:100]}...")
        else:
            print(f"❌ Failed: {result1.error}")
        
        print()
        
        # ========== Test 2: Screenshot Mode (Full Loading) ==========
        print("📸 Test 2: Screenshot Mode (Full Visual Loading)")
        print("-" * 80)
        
        result2 = await engine.execute_sync(
            "Take a screenshot of example.com"
        )
        
        if result2.success:
            print(f"✅ Success in {result2.elapsed}s (SCREENSHOT mode)")
            print(f"   Result: {result2.result[:100]}...")
        else:
            print(f"❌ Failed: {result2.error}")
        
        print()
        
        # ========== Test 3: Data Extraction Mode ==========
        print("📊 Test 3: Data Extraction Mode (Balanced Loading)")
        print("-" * 80)
        
        result3 = await engine.execute_sync(
            "Extract the main heading text from example.com"
        )
        
        if result3.success:
            print(f"✅ Success in {result3.elapsed}s (DATA mode)")
            print(f"   Result: {result3.result[:100]}...")
        else:
            print(f"❌ Failed: {result3.error}")
        
        print()
        
        # ========== Test 4: Second Request (Browser Reuse) ==========
        print("⚡ Test 4: Second Request (Browser Reuse Benefit)")
        print("-" * 80)
        
        result4 = await engine.execute_sync(
            "Navigate to wikipedia.org"
        )
        
        if result4.success:
            print(f"✅ Success in {result4.elapsed}s (Browser already initialized!)")
            print(f"   Result: {result4.result[:100]}...")
        else:
            print(f"❌ Failed: {result4.error}")
        
        print()
        
        # ========== Statistics ==========
        print("=" * 80)
        print("📊 Performance Statistics")
        print("=" * 80)
        
        stats = engine.get_stats()
        
        print(f"Total Queries:        {stats['total_queries']}")
        print(f"Successful Queries:   {stats['successful_queries']}")
        print(f"Success Rate:         {stats['success_rate']}%")
        print(f"Avg Response Time:    {stats['avg_response_time']}s")
        print(f"Active Sessions:      {stats['active_sessions']}")
        print(f"Uptime:              {stats['uptime_seconds']}s")
        print()
        
        print("🎯 Optimizations Enabled:")
        opts = stats['optimizations']
        print(f"  - Browser Reuse:      {'✅' if opts['browser_reuse'] else '❌'}")
        print(f"  - Resource Blocking:  {'✅' if opts['resource_blocking'] else '❌'}")
        print(f"  - Session Caching:    {'✅' if opts['session_caching'] else '❌'}")
        print()
        
        if 'performance_metrics' in stats:
            print("⏱️  Performance Metrics:")
            metrics = stats['performance_metrics']
            for operation, data in metrics.items():
                print(f"  {operation}:")
                print(f"    Count:   {data['count']}")
                print(f"    Avg:     {data['avg']}s")
                print(f"    Min:     {data['min']}s")
                print(f"    Max:     {data['max']}s")
                print(f"    P95:     {data['p95']}s")
        
        print()
        print("=" * 80)
        print("✨ Demo Complete!")
        print("=" * 80)


async def compare_modes():
    """Compare different resource strategies."""
    
    print()
    print("=" * 80)
    print("🔬 Resource Strategy Comparison")
    print("=" * 80)
    print()
    
    test_url = "https://example.com"
    
    strategies = ["interact", "screenshot", "data", "speed"]
    
    for strategy in strategies:
        config = BrowsyConfig(
            openai_api_key="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",
            openai_base_url="https://openrouter.ai/api/v1",
            openai_model="openai/gpt-4o-mini",
            playwright_headless=True,
            resource_strategy=strategy,  # Force specific strategy
        )
        
        async with BrowsyEngine(config) as engine:
            print(f"Testing {strategy.upper()} mode...")
            
            result = await engine.execute_sync(
                f"Go to {test_url}"
            )
            
            if result.success:
                print(f"  ✅ {strategy.upper()}: {result.elapsed}s")
            else:
                print(f"  ❌ {strategy.upper()}: Failed")
        
        print()


if __name__ == "__main__":
    # Run main demo
    asyncio.run(main())
    
    # Uncomment to run strategy comparison
    # asyncio.run(compare_modes())
