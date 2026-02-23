# Changelog

All notable changes to Browsy Automation will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-23

### Added
- Initial release of Browsy Automation
- Core `BrowsyEngine` for natural language web automation
- Configuration management via `BrowsyConfig`
- Support for multiple config sources (env vars, files, constructor args)
- Progress streaming via async generators
- Session management for conversation history
- FastAPI integration with SSE streaming endpoints
- CLI tool with commands: query, serve, config, info
- Comprehensive documentation and examples
- Support for Python 3.9+
- Optional dependencies for API and CLI features

### Features
- Natural language task execution via OpenAI LLM
- Playwright-powered browser automation
- MCP (Model Context Protocol) integration
- Real-time progress updates
- Context-aware multi-step automation
- Statistics and monitoring
- Async/await throughout
- Context manager support
- Type hints and Pydantic models

### Documentation
- README with quick start and examples
- Example scripts for basic usage, FastAPI integration, standalone server
- API reference documentation
- Configuration guide
- CLI usage examples

## [Unreleased]

### Planned
- TypeScript/npm client package
- React hooks for frontend integration
- More comprehensive test coverage
- Performance optimizations
- Additional browser options (Firefox, WebKit)
- Rate limiting and quota management
- Persistent session storage (Redis, SQLite)
- Batch query execution
- Query templates and presets
- Enhanced error handling and retries
- Metrics and telemetry integration
- Docker container support
- CI/CD pipeline

---

[0.1.0]: https://github.com/yourusername/browsy-automation/releases/tag/v0.1.0
