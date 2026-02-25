"""
Browsy Automation - Minimal Streamlit Frontend
==============================================

Clean frontend for Browsy web automation with live browser updates.
Headless mode enabled - see live browser actions without opening a window!

Run with: streamlit run streamlit_app.py
"""

import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict
import os

try:
    import streamlit as st
except ImportError:
    print("❌ Streamlit not installed.")
    print("Install with: pip install streamlit")
    exit(1)

from browsy import BrowsyEngine, BrowsyConfig
from browsy.types import EventType, ProgressStage


# ============ Configuration ============

# Configure your API key and base URL here
DEFAULT_API_KEY = "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"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


# ============ Page Config ============

st.set_page_config(
    page_title="Browsy Automation",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============ Custom CSS ============

st.markdown("""
<style>
    /* Clean minimal styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .browser-feed {
        background: #f8f9fa;
        border-left: 3px solid #1E88E5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-family: 'Monaco', 'Courier New', monospace;
        font-size: 0.9rem;
    }
    .action-step {
        padding: 0.75rem;
        margin: 0.3rem 0;
        border-radius: 4px;
        background: white;
        border-left: 3px solid #4CAF50;
    }
    .result-box {
        background: #E8F5E9;
        border: 1px solid #4CAF50;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #FFEBEE;
        border: 1px solid #F44336;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============ Session State Initialization ============

if "browser_actions" not in st.session_state:
    st.session_state.browser_actions = []

if "current_stage" not in st.session_state:
    st.session_state.current_stage = None

if "is_executing" not in st.session_state:
    st.session_state.is_executing = False

if "result" not in st.session_state:
    st.session_state.result = None

if "error" not in st.session_state:
    st.session_state.error = None

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "should_execute" not in st.session_state:
    st.session_state.should_execute = False

if "executing_query" not in st.session_state:
    st.session_state.executing_query = ""

if "screenshots" not in st.session_state:
    st.session_state.screenshots = []


# ============ Helper Functions ============

def add_browser_action(action: str, icon: str = "🔹"):
    """Add a browser action to the live feed."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.browser_actions.append({
        "time": timestamp,
        "action": action,
        "icon": icon
    })


def reset_state():
    """Reset execution state."""
    st.session_state.browser_actions = []
    st.session_state.current_stage = None
    st.session_state.result = None
    st.session_state.error = None
    st.session_state.screenshots = []


def cleanup_automation_processes():
    """Thoroughly clean up stale Playwright/Node automation processes.
    
    This kills only automation-related processes (Playwright MCP node servers
    and automation browser instances), NOT the user's own browser.
    """
    import subprocess
    
    # Step 1: Kill node.exe processes that are running Playwright MCP
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" "
            "| Where-Object { $_.CommandLine -match 'playwright' } "
            "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            capture_output=True, timeout=10
        )
    except Exception:
        # Fallback: kill all node.exe
        try:
            subprocess.run(
                ['taskkill', '/F', '/IM', 'node.exe', '/T'],
                capture_output=True, shell=True, timeout=5
            )
        except Exception:
            pass
    
    # Step 2: Kill ONLY automation browser instances (identified by --remote-debugging or --headless flags)
    # This does NOT kill the user's normal browser
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Process "
            "| Where-Object { $_.Name -match 'chrom' -and $_.CommandLine -match 'remote-debugging|--headless|--disable-background' } "
            "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.run(
            ['powershell', '-NoProfile', '-Command', ps_cmd],
            capture_output=True, timeout=10
        )
    except Exception:
        pass
    
    # Step 3: Wait for OS to fully release resources (ports, file locks, etc.)
    time.sleep(2)


async def execute_query_async(query: str, config: BrowsyConfig, 
                               progress_bar, stage_container, actions_container):
    """Execute query with live browser action updates."""
    st.session_state.is_executing = True
    start_time = time.time()
    engine = None
    
    try:
        browser_mode = "headed mode (visible window)" if not config.playwright_headless else "headless mode"
        add_browser_action(f"🚀 Starting Browsy Engine in {browser_mode}...", "🚀")
        
        # ALWAYS clean up stale automation processes before starting a new instance
        add_browser_action("🧹 Cleaning stale automation processes (not your browser)...", "🧹")
        cleanup_automation_processes()
        
        # Initialize the engine with retry logic
        engine = BrowsyEngine(config)
        add_browser_action("⚙️ Initializing browser engine...", "⚙️")
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                await engine.initialize()
                break
            except Exception as init_err:
                if attempt < max_retries:
                    add_browser_action(f"⚠️ Init attempt {attempt + 1} failed, retrying...", "⚠️")
                    await engine.cleanup()
                    cleanup_automation_processes()
                    engine = BrowsyEngine(config)
                else:
                    raise init_err
        
        add_browser_action("✅ Engine ready!", "✅")
        
        if not config.playwright_headless:
            add_browser_action("👁️ Watch for the browser window to open...", "👁️")
        
        try:
            async for event in engine.execute(query):
                
                if event.type == EventType.PROGRESS:
                    # Update progress
                    progress_bar.progress(event.progress / 100, text=f"{event.progress}% - {event.stage.value}")
                    st.session_state.current_stage = event.stage.value
                    
                    # Add to browser action feed
                    icon_map = {
                        "initializing": "⚙️",
                        "initialized": "✅",
                        "connecting": "🔗",
                        "connected": "🌐",
                        "processing": "⚡",
                        "completing": "🏁",
                        "complete": "✨"
                    }
                    icon = icon_map.get(event.stage.value, "🔹")
                    add_browser_action(event.message, icon)
                    
                    # Live update actions display
                    with actions_container:
                        browser_label = "Headless" if config.playwright_headless else "Visible Window"
                        st.markdown(f"### 🤖 Live Browser Actions ({browser_label})")
                        for action in st.session_state.browser_actions[-10:]:  # Show last 10
                            st.markdown(
                                f'<div class="action-step">'
                                f'<code>{action["time"]}</code> {action["icon"]} {action["action"]}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    
                elif event.type == EventType.RESULT:
                    elapsed = time.time() - start_time
                    st.session_state.result = event.result
                    
                    # Capture screenshots from the event
                    if hasattr(event, 'screenshots') and event.screenshots:
                        st.session_state.screenshots = event.screenshots
                        add_browser_action(f"📸 {len(event.screenshots)} screenshot(s) captured", "📸")
                    
                    add_browser_action(f"✅ Task completed in {elapsed:.2f}s", "✅")
                    
                    # Add to history
                    st.session_state.query_history.insert(0, {
                        "query": query,
                        "result": event.result[:200] + "..." if len(event.result) > 200 else event.result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed": elapsed,
                        "screenshot_count": len(event.screenshots) if hasattr(event, 'screenshots') else 0,
                    })
                    
                elif event.type == EventType.ERROR:
                    st.session_state.error = event.message
                    add_browser_action(f"❌ Error: {event.message}", "❌")
                    
                    # Add helpful troubleshooting for API errors
                    if "api" in event.message.lower() or "key" in event.message.lower():
                        st.session_state.error += "\n\n**💡 API Troubleshooting:**"
                        st.session_state.error += "\n- Check your API key in the sidebar"
                        st.session_state.error += "\n- Verify base URL is correct for your provider"
                        st.session_state.error += f"\n- Current: {config.openai_base_url or 'OpenAI default'}"
                        st.session_state.error += f"\n- Model: {config.openai_model}"
                    elif "quota" in event.message.lower() or "limit" in event.message.lower():
                        st.session_state.error += "\n\n**💡 Quota Issue:**"
                        st.session_state.error += "\n- Your API quota may be exceeded"
                        st.session_state.error += "\n- Check your account balance/credits"
                        st.session_state.error += "\n- Try a different model or API provider"
        
        finally:
            # Clean up engine resources
            add_browser_action("🧹 Cleaning up resources...", "🧹")
            if engine:
                await engine.cleanup()
            # Force-kill any remaining automation processes
            cleanup_automation_processes()
            add_browser_action("✅ Cleanup complete — browser fully closed", "✅")
                    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        st.session_state.error = error_msg
        add_browser_action(f"❌ Exception: {error_msg}", "❌")
        
        # Detailed troubleshooting based on error type
        if "api" in str(e).lower() or "openai" in str(e).lower():
            st.session_state.error += "\n\n**💡 API Configuration Issue:**"
            st.session_state.error += f"\n- API Key configured: {'✅ Yes' if config.openai_api_key else '❌ No'}"
            st.session_state.error += f"\n- Base URL: {config.openai_base_url or 'Default OpenAI'}"
            st.session_state.error += f"\n- Model: {config.openai_model}"
            st.session_state.error += "\n\n**Suggestions:**"
            st.session_state.error += "\n1. Verify API key is valid"
            st.session_state.error += "\n2. Check if model name is correct"
            st.session_state.error += "\n3. Ensure you have credits/quota available"
            st.session_state.error += "\n4. Try a different model (e.g., 'openai/gpt-3.5-turbo')"
        elif "playwright" in str(e).lower():
            st.session_state.error += "\n\n**💡 Browser Issue:**"
            st.session_state.error += "\n- Run: `playwright install` to install browsers"
            st.session_state.error += "\n- Click 'Reset Browser Processes' button"
        elif "mcp" in str(e).lower():
            st.session_state.error += "\n\n**💡 MCP Issue:**"
            st.session_state.error += "\n- Ensure mcp-agent is installed: `pip install mcp-agent`"
            st.session_state.error += "\n- Try restarting the server"
        
        # Log full error for debugging
        import traceback
        full_error = traceback.format_exc()
        add_browser_action(f"📋 Full error logged (expand for details)", "📋")
        print(f"\n{'='*50}\nFULL ERROR:\n{full_error}\n{'='*50}\n")
    
    finally:
        st.session_state.is_executing = False


def run_query(query: str, config: BrowsyConfig, progress_bar, stage_container, actions_container):
    """Synchronous wrapper to run async query."""
    reset_state()
    
    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async function
    loop.run_until_complete(
        execute_query_async(query, config, progress_bar, stage_container, actions_container)
    )



# ============ Minimal Sidebar Configuration ============

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    with st.expander("🔑 API Configuration", expanded=False):
        api_key = st.text_input(
            "API Key",
            value=DEFAULT_API_KEY,
            type="password",
            help="Your LLM API key"
        )
        
        base_url = st.text_input(
            "Base URL",
            value=DEFAULT_BASE_URL,
            help="API endpoint"
        )
        
        model = st.text_input(
            "Model",
            value=DEFAULT_MODEL,
            help="Model name"
        )
        
        # Test API button
        if st.button("🧪 Test API Connection", use_container_width=True):
            with st.spinner("Testing API connection..."):
                import requests
                try:
                    response = requests.post(
                        f"{base_url or 'https://api.openai.com/v1'}/chat/completions",
                        headers={
                            'Authorization': f'Bearer {api_key}',
                            'Content-Type': 'application/json'
                        },
                        json={
                            'model': model,
                            'messages': [{'role': 'user', 'content': 'Test'}],
                            'max_tokens': 10
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        st.success("✅ API connection successful!")
                        data = response.json()
                        if 'usage' in data:
                            st.info(f"📊 Tokens used: {data['usage'].get('total_tokens', 'N/A')}")
                    else:
                        st.error(f"❌ API Error {response.status_code}")
                        st.code(response.text[:500])
                        
                except Exception as e:
                    st.error(f"❌ Connection failed: {e}")
                    st.info("💡 Check your API key, base URL, and internet connection")
        
        # Check quota for OpenRouter
        if "openrouter" in (base_url or "").lower():
            if st.button("💰 Check OpenRouter Quota", use_container_width=True):
                with st.spinner("Checking quota..."):
                    import requests
                    try:
                        response = requests.get(
                            "https://openrouter.ai/api/v1/auth/key",
                            headers={'Authorization': f'Bearer {api_key}'},
                            timeout=5
                        )
                        if response.status_code == 200:
                            data = response.json().get('data', {})
                            limit = data.get('limit', 'N/A')
                            remaining = data.get('limit_remaining', 'N/A')
                            st.success(f"💰 Quota: ${remaining:.2f} / ${limit} remaining")
                            
                            if isinstance(remaining, (int, float)) and remaining < 0.5:
                                st.warning("⚠️ Low quota! Consider adding credits.")
                        else:
                            st.error(f"❌ Could not check quota: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Quota check failed: {e}")
    
    with st.expander("🌐 Browser Settings", expanded=True):
        headless = st.checkbox(
            "Headless Mode (Background)",
            value=True,
            help="✅ ON = Browser runs invisibly in background\n❌ OFF = See browser window in real-time!"
        )
        
        if not headless:
            st.success("🎥 **Real-Time Mode**: A NEW browser window will open!")
            st.info("💡 **Important**: The automation uses a SEPARATE browser process, not this one")
            st.caption("When you see the automation window, it will be a fresh browser instance")
        else:
            st.info("👻 **Headless Mode**: Browser runs invisibly")
        
        # Reset browser button - more prominent
        st.markdown("---")
        st.markdown("**🔧 If Browser Won't Open:**")
        if st.button("🔄 Reset Automation Processes Only", 
                    help="Kill only automation processes (Node.js/Playwright servers). Won't close this browser!",
                    use_container_width=True,
                    type="secondary"):
            with st.spinner("Resetting automation processes..."):
                import subprocess
                try:
                    # Only kill node.exe processes (Playwright MCP servers)
                    # This will NOT kill your browser viewing this page!
                    result = subprocess.run(
                        ["taskkill", "/F", "/IM", "node.exe", "/T"], 
                        capture_output=True, 
                        shell=True, 
                        timeout=5
                    )
                    time.sleep(1)
                    
                    st.success("✅ Automation processes reset!")
                    st.info("💡 Now try your task - a NEW browser window will open (separate from this one)")
                    
                except Exception as e:
                    st.error(f"❌ Reset failed: {e}")
                    st.info("💡 You can also manually close 'node.exe' processes in Task Manager")
    
    # Configure playwright args - remove --headless flag when in headed mode
    playwright_args = ["@playwright/mcp@latest"]
    if headless:
        playwright_args.append("--headless")
    
    # Headless mode based on user selection
    # IMPORTANT: Disable ALL browser pooling/reuse to force fresh instances
    config = BrowsyConfig(
        openai_api_key=api_key,
        openai_base_url=base_url if base_url else None,
        openai_model=model,
        playwright_headless=headless,  # User-controlled
        playwright_args=playwright_args,  # Dynamic args based on headless mode
        use_history=True,
        max_tokens=10000,
        log_level="info",  # Must be lowercase: 'debug', 'info', 'warning', or 'error'
        enable_browser_reuse=False,  # Always disable - forces fresh browser instance
        enable_resource_blocking=False,  # Disable to avoid conflicts
        enable_session_caching=False,  # Disable - fresh browser every time
        max_browser_pool_size=1,  # Only one browser at a time
        enable_performance_metrics=False,  # Disable for simplicity
    )
    
    st.markdown("---")
    
    # Query History
    st.markdown("## 📜 Recent Queries")
    
    if st.session_state.query_history:
        for idx, item in enumerate(st.session_state.query_history[:3]):  # Show last 3
            with st.expander(f"🕐 {item['timestamp']}", expanded=False):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Time:** {item['elapsed']:.2f}s")
                if st.button("🔄 Run Again", key=f"rerun_{idx}"):
                    st.session_state.query_to_run = item['query']
                    st.rerun()
    else:
        st.caption("No queries yet")


# ============ Main Content ============

# Header
st.markdown('<div class="main-title">🌐 Browsy Automation</div>', unsafe_allow_html=True)

browser_mode_text = "headless browser tracking" if config.playwright_headless else "real-time browser window"
st.markdown(
    f'<p class="subtitle">AI-powered web automation with {browser_mode_text}</p>',
    unsafe_allow_html=True
)

# Validation check
if not config.validate_api_key():
    st.error("❌ Invalid API key! Please configure in the sidebar.")
    st.stop()

# Query Input Section
st.markdown("### 💬 Enter Your Task")

query_input = st.text_area(
    "What would you like the browser to do?",
    placeholder="e.g., Navigate to github.com and get the top 5 trending repositories",
    value=st.session_state.get("query_to_run", ""),
    height=100,
    disabled=st.session_state.is_executing,
    label_visibility="collapsed"
)

# Clear pre-filled query after it's displayed
if "query_to_run" in st.session_state:
    del st.session_state.query_to_run

# Important notice for headed mode
if not config.playwright_headless and not st.session_state.is_executing:
    st.info(
        "👁️ **Headed Mode Active**: When you click Execute, a NEW Chrome window will open "
        "(separate from this browser). The automation will run in that window, not this one."
    )


# Execute Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    browser_icon = "👻" if config.playwright_headless else "🎥"
    button_text = f"{browser_icon} Execute Task" if not st.session_state.is_executing else "⏳ Running..."
    
    if st.button(
        button_text,
        disabled=st.session_state.is_executing or not query_input,
        use_container_width=True,
        type="primary"
    ):
        st.session_state.should_execute = True
        st.session_state.executing_query = query_input

# Quick Example Queries
with st.expander("💡 Example Queries", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📰 GitHub Trending", disabled=st.session_state.is_executing, use_container_width=True):
            st.session_state.query_to_run = "Navigate to github.com and get the top 5 trending repositories"
            st.rerun()
        
        if st.button("🔍 Wikipedia Search", disabled=st.session_state.is_executing, use_container_width=True):
            st.session_state.query_to_run = "Go to wikipedia.org and search for Python programming language"
            st.rerun()
    
    with col2:
        if st.button("📊 Page Title", disabled=st.session_state.is_executing, use_container_width=True):
            st.session_state.query_to_run = "Navigate to example.com and get the page title"
            st.rerun()
        
        if st.button("🌐 News Headlines", disabled=st.session_state.is_executing, use_container_width=True):
            st.session_state.query_to_run = "Go to news.ycombinator.com and get the top 5 headlines"
            st.rerun()

st.markdown("---")

# Execution Section
if st.session_state.should_execute and st.session_state.executing_query:
    
    # Create placeholders for live updates
    progress_container = st.empty()
    stage_container = st.empty()
    actions_container = st.container()
    
    # Execute with live updates
    spinner_text = "Initializing browser..." if not config.playwright_headless else "Initializing headless browser..."
    with st.spinner(spinner_text):
        try:
            with progress_container:
                progress_bar = st.progress(0, text="Starting...")
            
            run_query(
                st.session_state.executing_query,
                config,
                progress_bar,
                stage_container,
                actions_container
            )
            
        except Exception as e:
            st.session_state.error = f"Execution error: {str(e)}"
            add_browser_action(f"❌ Fatal error: {str(e)}", "❌")
    
    # Clear flag
    st.session_state.should_execute = False
    st.session_state.executing_query = ""
    st.rerun()

# Display Browser Action Feed if available
if st.session_state.browser_actions:
    with st.expander("🤖 Browser Action Log", expanded=True):
        for action in st.session_state.browser_actions:
            st.markdown(
                f'<div class="action-step">'
                f'<code>{action["time"]}</code> {action["icon"]} {action["action"]}'
                f'</div>',
                unsafe_allow_html=True
            )

# Display Result
if st.session_state.result:
    # Display Screenshots FIRST (visual proof at the top)
    if st.session_state.screenshots:
        st.markdown("### 📸 Browser Screenshots")
        import base64
        for idx, screenshot in enumerate(st.session_state.screenshots):
            try:
                # Handle both raw base64 strings and data-URI prefixed strings
                raw_data = screenshot["data"]
                # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
                if "," in raw_data and raw_data.startswith("data:"):
                    raw_data = raw_data.split(",", 1)[1]
                image_bytes = base64.b64decode(raw_data)
                mime = screenshot.get("mime_type", "image/png")
                tool = screenshot.get("tool_name", "screenshot")
                
                st.image(
                    image_bytes,
                    caption=f"Screenshot {idx + 1} — {tool}",
                    use_container_width=True,
                )
                
                # Download button for each screenshot
                st.download_button(
                    label=f"💾 Download Screenshot {idx + 1}",
                    data=image_bytes,
                    file_name=f"browsy_screenshot_{idx + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime=mime,
                    key=f"dl_screenshot_{idx}",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Could not display screenshot {idx + 1}: {e}")
                # Show raw data length for debugging
                st.caption(f"Debug: data length = {len(screenshot.get('data', ''))}, mime = {screenshot.get('mime_type', 'unknown')}")
    
    # Display text result with proper markdown rendering
    st.markdown("### ✅ Result")
    st.markdown(st.session_state.result)
    
    # Download result text
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.download_button(
            label="💾 Download Result Text",
            data=st.session_state.result,
            file_name=f"browsy_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

# Display Error
if st.session_state.error:
    st.markdown("### ❌ Error")
    st.markdown(
        f'<div class="error-box">{st.session_state.error}</div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #999; font-size: 0.9rem;">'
    '🌐 Browsy Automation | Web Automation with Headless or Real-Time Browser Mode<br>'
    'Powered by Playwright & OpenAI'
    '</div>',
    unsafe_allow_html=True
)
