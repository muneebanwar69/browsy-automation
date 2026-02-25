"""
Performance optimization utilities for Browsy.

This module provides:
- Browser instance pooling and reuse
- Task-aware resource blocking strategies
- Session and cookie caching
- Performance metrics tracking
- Smart wait strategies
"""

from __future__ import annotations

import time
import hashlib
import asyncio
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field


class ResourceStrategy(str, Enum):
    """Resource loading strategies based on task type."""
    SCREENSHOT = "screenshot"      # Load everything for visual accuracy
    DATA = "data"                  # Block images, keep CSS for layout
    INTERACTION = "interact"       # Minimal resources, just functionality
    SPEED = "speed"               # Block everything possible


class TaskType(str, Enum):
    """Detected task types from user queries."""
    SCREENSHOT = "screenshot"
    DATA_EXTRACTION = "data_extraction"
    INTERACTION = "interaction"
    NAVIGATION = "navigation"
    UNKNOWN = "unknown"


@dataclass
class PerformanceTiming:
    """Track timing for various operations."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    def complete(self):
        """Mark operation as complete and calculate duration."""
        self.end_time = time.time()
        self.duration = round(self.end_time - self.start_time, 3)


class PerformanceMetrics:
    """Track and analyze performance metrics."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.active_operations: Dict[str, PerformanceTiming] = {}
    
    def start_operation(self, operation: str) -> str:
        """Start tracking an operation. Returns tracking ID."""
        tracking_id = f"{operation}_{time.time()}"
        self.active_operations[tracking_id] = PerformanceTiming(
            operation=operation,
            start_time=time.time()
        )
        return tracking_id
    
    def end_operation(self, tracking_id: str):
        """End tracking for an operation."""
        if tracking_id in self.active_operations:
            timing = self.active_operations.pop(tracking_id)
            timing.complete()
            
            # Record metrics
            if timing.operation not in self.timings:
                self.timings[timing.operation] = []
                self.operation_counts[timing.operation] = 0
            
            self.timings[timing.operation].append(timing.duration)
            self.operation_counts[timing.operation] += 1
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation."""
        if operation not in self.timings or not self.timings[operation]:
            return None
        
        times = self.timings[operation]
        sorted_times = sorted(times)
        
        return {
            'count': len(times),
            'avg': round(sum(times) / len(times), 3),
            'min': round(min(times), 3),
            'max': round(max(times), 3),
            'median': round(sorted_times[len(sorted_times) // 2], 3),
            'p95': round(sorted_times[int(len(sorted_times) * 0.95)], 3) if len(times) > 1 else round(times[0], 3),
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {
            op: self.get_stats(op)
            for op in self.timings.keys()
            if self.get_stats(op) is not None
        }


class CacheManager:
    """Manage caching for pages, sessions, and selectors."""
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            ttl: Time-to-live for cache entries in seconds
        """
        self.ttl = ttl
        self.page_cache: Dict[str, tuple[float, Any]] = {}
        self.login_sessions: Dict[str, Dict[str, Any]] = {}
        self.element_selectors: Dict[str, Dict[str, str]] = {}
    
    def cache_key(self, url: str, params: Optional[str] = None) -> str:
        """Generate cache key from URL and params."""
        data = f"{url}:{params or ''}"
        return hashlib.md5(data.encode()).hexdigest()
    
    async def get_or_fetch(
        self,
        url: str,
        fetch_fn: Callable,
        params: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or fetch if not cached/expired.
        
        Args:
            url: URL key
            fetch_fn: Async function to fetch data
            params: Optional parameters for cache key
            ttl: Optional custom TTL for this entry
        """
        key = self.cache_key(url, params)
        ttl = ttl or self.ttl
        
        if key in self.page_cache:
            cached_time, content = self.page_cache[key]
            if time.time() - cached_time < ttl:
                return content
        
        # Fetch and cache
        content = await fetch_fn()
        self.page_cache[key] = (time.time(), content)
        return content
    
    def save_login_session(self, site: str, cookies: List[Dict[str, Any]], storage_state: Optional[Dict] = None):
        """
        Save authenticated session for a site.
        
        Args:
            site: Site identifier (domain)
            cookies: Browser cookies
            storage_state: Optional browser storage state
        """
        self.login_sessions[site] = {
            'cookies': cookies,
            'storage_state': storage_state,
            'timestamp': time.time(),
        }
    
    async def restore_login_session(self, site: str, context) -> bool:
        """
        Restore saved login session if valid.
        
        Args:
            site: Site identifier
            context: Playwright browser context
            
        Returns:
            True if session was restored, False otherwise
        """
        if site not in self.login_sessions:
            return False
        
        session = self.login_sessions[site]
        
        # Check if session is still valid
        if time.time() - session['timestamp'] > self.ttl:
            # Session expired
            del self.login_sessions[site]
            return False
        
        try:
            # Restore cookies
            await context.add_cookies(session['cookies'])
            
            # Restore storage state if available
            if session.get('storage_state'):
                # Note: Storage state is typically set during context creation
                # This is more for future reference
                pass
            
            return True
        except Exception as e:
            print(f"Warning: Failed to restore session for {site}: {e}")
            return False
    
    def clear_expired(self):
        """Clear expired cache entries."""
        current_time = time.time()
        
        # Clear page cache
        expired_pages = [
            key for key, (cached_time, _) in self.page_cache.items()
            if current_time - cached_time > self.ttl
        ]
        for key in expired_pages:
            del self.page_cache[key]
        
        # Clear login sessions
        expired_sessions = [
            site for site, session in self.login_sessions.items()
            if current_time - session['timestamp'] > self.ttl
        ]
        for site in expired_sessions:
            del self.login_sessions[site]


class TaskDetector:
    """Detect task type from user queries using keyword analysis."""
    
    # Keywords for different task types
    SCREENSHOT_KEYWORDS = [
        'screenshot', 'capture', 'image', 'visual', 'picture', 'photo',
        'how does it look', 'show me', 'appearance', 'display'
    ]
    
    DATA_KEYWORDS = [
        'extract', 'scrape', 'get data', 'find text', 'read', 'parse',
        'collect', 'gather', 'fetch data', 'retrieve', 'download data',
        'list all', 'get all', 'find all'
    ]
    
    INTERACTION_KEYWORDS = [
        'click', 'login', 'fill', 'submit', 'type', 'enter', 'press',
        'select', 'choose', 'check', 'uncheck', 'upload', 'sign in',
        'log in', 'button', 'form'
    ]
    
    NAVIGATION_KEYWORDS = [
        'navigate', 'go to', 'visit', 'open', 'load', 'browse to',
        'navigate to', 'access'
    ]
    
    @classmethod
    def detect_task_type(cls, query: str) -> TaskType:
        """
        Detect task type from query using keyword matching.
        
        Args:
            query: User query string
            
        Returns:
            Detected TaskType
        """
        query_lower = query.lower()
        
        # Check for screenshot (highest priority for accuracy)
        if any(word in query_lower for word in cls.SCREENSHOT_KEYWORDS):
            return TaskType.SCREENSHOT
        
        # Check for data extraction
        if any(word in query_lower for word in cls.DATA_KEYWORDS):
            return TaskType.DATA_EXTRACTION
        
        # Check for interaction
        if any(word in query_lower for word in cls.INTERACTION_KEYWORDS):
            return TaskType.INTERACTION
        
        # Check for navigation
        if any(word in query_lower for word in cls.NAVIGATION_KEYWORDS):
            return TaskType.NAVIGATION
        
        return TaskType.UNKNOWN
    
    @classmethod
    def get_resource_strategy(cls, task_type: TaskType) -> ResourceStrategy:
        """
        Get optimal resource loading strategy for task type.
        
        Args:
            task_type: Detected task type
            
        Returns:
            ResourceStrategy for the task
        """
        strategy_map = {
            TaskType.SCREENSHOT: ResourceStrategy.SCREENSHOT,
            TaskType.DATA_EXTRACTION: ResourceStrategy.DATA,
            TaskType.INTERACTION: ResourceStrategy.INTERACTION,
            TaskType.NAVIGATION: ResourceStrategy.INTERACTION,
            TaskType.UNKNOWN: ResourceStrategy.INTERACTION,  # Default to fast
        }
        return strategy_map.get(task_type, ResourceStrategy.INTERACTION)


class BrowserOptimizer:
    """Optimize browser page loading based on task requirements."""
    
    # Tracking domains to block
    TRACKING_DOMAINS = [
        'google-analytics.com',
        'googletagmanager.com',
        'doubleclick.net',
        'facebook.com/tr',
        'facebook.net',
        'analytics',
        '/ads/',
        'advertising',
        'track',
        'metrics',
        'telemetry',
    ]
    
    @staticmethod
    async def configure_page(
        page,
        strategy: ResourceStrategy = ResourceStrategy.INTERACTION,
    ) -> tuple:
        """
        Configure page with optimal resource loading for the task.
        
        Args:
            page: Playwright page instance
            strategy: Resource loading strategy
            
        Returns:
            Tuple of (page, wait_strategy, timeout)
        """
        if strategy == ResourceStrategy.SCREENSHOT:
            # Load CSS & images for accurate screenshots, block only tracking
            await page.route("**/*", lambda route: (
                route.abort() if any(
                    domain in route.request.url.lower()
                    for domain in BrowserOptimizer.TRACKING_DOMAINS
                ) else route.continue_()
            ))
            wait_strategy = "load"  # Wait for full visual render
            timeout = 30000  # 30s for complete load
            
        elif strategy == ResourceStrategy.DATA:
            # Keep CSS for layout, block images/fonts
            await page.route("**/*", lambda route: (
                route.abort() if (
                    route.request.resource_type in ["image", "media", "font"]
                    or any(
                        domain in route.request.url.lower()
                        for domain in BrowserOptimizer.TRACKING_DOMAINS
                    )
                ) else route.continue_()
            ))
            wait_strategy = "domcontentloaded"
            timeout = 15000  # 15s
            
        elif strategy == ResourceStrategy.INTERACTION:
            # Minimal resources for interactions
            await page.route("**/*", lambda route: (
                route.abort() if (
                    route.request.resource_type in ["image", "stylesheet", "font", "media"]
                    or any(
                        domain in route.request.url.lower()
                        for domain in BrowserOptimizer.TRACKING_DOMAINS
                    )
                ) else route.continue_()
            ))
            wait_strategy = "domcontentloaded"
            timeout = 10000  # 10s
            
        else:  # SPEED mode
            # Maximum speed - block everything non-essential
            ALLOWED_TYPES = ["document", "script", "xhr", "fetch"]
            await page.route("**/*", lambda route: (
                route.continue_() if route.request.resource_type in ALLOWED_TYPES
                else route.abort()
            ))
            wait_strategy = "domcontentloaded"
            timeout = 8000  # 8s
        
        # Set timeouts
        page.set_default_timeout(timeout)
        page.set_default_navigation_timeout(timeout)
        
        return page, wait_strategy, timeout
    
    @staticmethod
    async def smart_navigate(
        page,
        url: str,
        strategy: ResourceStrategy = ResourceStrategy.INTERACTION,
    ):
        """
        Navigate with optimal settings for the task.
        
        Args:
            page: Playwright page
            url: URL to navigate to
            strategy: Resource loading strategy
        """
        page, wait_strategy, timeout = await BrowserOptimizer.configure_page(page, strategy)
        
        try:
            await page.goto(url, wait_until=wait_strategy, timeout=timeout)
        except Exception as e:
            # If navigation fails, try with more lenient settings
            print(f"Navigation with {strategy} failed, retrying with lenient settings: {e}")
            try:
                await page.goto(url, wait_until='commit', timeout=timeout)
            except:
                # If still fails, let it propagate
                raise


class AdaptivePageLoader:
    """Learn and adapt resource loading per site."""
    
    def __init__(self):
        self.site_profiles: Dict[str, Dict[str, Any]] = {}
    
    def get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        return urlparse(url).netloc
    
    async def smart_load(
        self,
        page,
        url: str,
        task_type: TaskType,
    ):
        """
        Adaptively load page based on learned profile or task type.
        
        Args:
            page: Playwright page
            url: URL to load
            task_type: Type of task being performed
        """
        domain = self.get_domain(url)
        cache_key = f"{domain}:{task_type.value}"
        
        # Use learned profile if available
        if cache_key in self.site_profiles:
            profile = self.site_profiles[cache_key]
            # Apply learned optimizations
            # (Future enhancement: use historical data for better blocking)
        
        # For now, use task-based strategy
        strategy = TaskDetector.get_resource_strategy(task_type)
        await BrowserOptimizer.smart_navigate(page, url, strategy)
        
        # Learn from this load (future enhancement)
        # Track which resources were actually used


class BrowserPool:
    """Manage pool of reusable browser instances."""
    
    def __init__(self, max_browsers: int = 1):
        """
        Initialize browser pool.
        
        Args:
            max_browsers: Maximum number of browser instances to maintain
        """
        self.max_browsers = max_browsers
        self.browsers: List[Any] = []
        self.contexts: List[Any] = []
        self.pages: List[Any] = []
        self.playwright = None
        self.in_use: Dict[int, bool] = {}
    
    async def initialize(self, headless: bool = True):
        """
        Initialize Playwright and browser pool.
        
        Args:
            headless: Whether to run browsers in headless mode
        """
        if self.playwright is None:
            from playwright.async_api import async_playwright
            self.playwright = await async_playwright().start()
        
        # Create initial browser if none exist
        if not self.browsers:
            browser = await self.playwright.chromium.launch(
                headless=headless,
                args=[
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-gpu',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                ]
            )
            self.browsers.append(browser)
            self.in_use[0] = False
    
    async def get_browser(self) -> tuple:
        """
        Get or create an available browser instance.
        
        Returns:
            Tuple of (browser_index, browser, context, page)
        """
        # Find available browser
        for idx, browser in enumerate(self.browsers):
            if not self.in_use.get(idx, False):
                self.in_use[idx] = True
                
                # Create new context and page
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    bypass_csp=True,
                )
                page = await context.new_page()
                
                return idx, browser, context, page
        
        # All browsers busy, create new one if under limit
        if len(self.browsers) < self.max_browsers:
            idx = len(self.browsers)
            browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-gpu',
                ]
            )
            self.browsers.append(browser)
            self.in_use[idx] = True
            
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                bypass_csp=True,
            )
            page = await context.new_page()
            
            return idx, browser, context, page
        
        # Wait for available browser
        while True:
            await asyncio.sleep(0.1)
            for idx, browser in enumerate(self.browsers):
                if not self.in_use.get(idx, False):
                    self.in_use[idx] = True
                    context = await browser.new_context()
                    page = await context.new_page()
                    return idx, browser, context, page
    
    async def release_browser(self, idx: int, context, page):
        """
        Release browser back to pool.
        
        Args:
            idx: Browser index
            context: Browser context to close
            page: Page to close
        """
        try:
            await page.close()
            await context.close()
        except:
            pass
        finally:
            self.in_use[idx] = False
    
    async def cleanup(self):
        """Clean up all browser instances."""
        for browser in self.browsers:
            try:
                await browser.close()
            except:
                pass
        
        if self.playwright:
            try:
                await self.playwright.stop()
            except:
                pass
        
        self.browsers = []
        self.in_use = {}
        self.playwright = None


# Singleton instances
_browser_pool: Optional[BrowserPool] = None
_cache_manager: Optional[CacheManager] = None
_performance_metrics: Optional[PerformanceMetrics] = None


def get_browser_pool() -> BrowserPool:
    """Get global browser pool instance."""
    global _browser_pool
    if _browser_pool is None:
        _browser_pool = BrowserPool()
    return _browser_pool


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def get_performance_metrics() -> PerformanceMetrics:
    """Get global performance metrics instance."""
    global _performance_metrics
    if _performance_metrics is None:
        _performance_metrics = PerformanceMetrics()
    return _performance_metrics
