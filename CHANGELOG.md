# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-11-05

### Changed - Replit Migration
- **BREAKING**: Migrated from keyring-based secret storage to Replit Secrets (environment variables)
- **BREAKING**: Removed `--reset-secrets` command-line flag (no longer needed with environment variables)
- Updated default port from 8080 to 5000 for Replit compatibility
- Configured Flask to run on 0.0.0.0:5000 for proper Replit routing
- Cleaned up requirements.txt and removed keyring dependency

### Added
- Replit deployment configuration using Gunicorn WSGI server (4 workers)
- Comprehensive replit.md documentation with setup instructions
- Workflow configuration for automatic bot execution in Replit
- Enhanced .gitignore with comprehensive Python patterns
- Clear warning messages when required secrets are not set
- README badges for Replit, Python, and Flask
- CHANGELOG.md for version tracking
- Production-ready deployment using VM configuration with Gunicorn

### Fixed
- **CRITICAL**: Background workers now start correctly in both development and production environments
  - Workers previously only started when running `python nft_sales_bot.py` directly
  - Fixed by moving worker initialization to module level (executes on import)
  - Ensures Gunicorn deployments have functional job queue for webhook processing

### Improved
- Documentation with Replit-specific setup instructions
- Secret management with clearer error messages
- Production deployment readiness with VM configuration
- Developer experience with better organized codebase
- Clear separation between development server (Flask) and production server (Gunicorn)
- Added explicit documentation about Flask vs Gunicorn usage

### Security
- Secrets now managed through Replit's encrypted secret storage
- No secrets stored in code or version control
- Environment variable-based configuration prevents accidental exposure

## [1.0.0] - 2024

### Initial Release
- NFT sales monitoring via Moralis webhooks
- Twitter/X posting with OAuth 1.0a authentication
- Single NFT sale posts with images and metadata
- Sweep purchase detection with collage generation
- Background job queue with worker threads
- IPFS multi-gateway support for image fetching
- Webhook signature verification with Keccak-256
- Idempotency tracking (in-memory and Redis options)
- Metrics endpoints (JSON and Prometheus formats)
- Debug endpoints for testing single and sweep posts
- Health and readiness check endpoints
- Collage generation with configurable layouts
- ETH to USD price conversion via CoinGecko
- OpenSea integration for NFT links
- Configurable collection monitoring
- Thread pools for parallel downloads and uploads
- Safe image handling with size limits and pixel bomb protection

### Features
- Support for up to 4 collages per tweet (12 images each)
- Adaptive text truncation for Twitter's 280 character limit
- Automatic retry logic for network requests
- Safe logging with secret redaction
- Flexible configuration via environment variables
- Production-ready with Gunicorn support

---

## Version History Summary

- **v2.0.0** - Replit migration with environment-based secrets
- **v1.0.0** - Initial release with keyring-based secrets
