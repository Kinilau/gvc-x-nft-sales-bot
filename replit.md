# NFT Sales Bot

## Overview

An automated NFT sales bot that monitors blockchain transactions via Moralis webhooks and posts updates to Twitter/X. The bot handles both single NFT sales and sweep purchases (bulk buys), creating attractive collages for multi-item sales.

**Current State**: Fully configured and running in Replit environment. The Flask server is listening on port 5000 and ready to receive webhook events from Moralis.

## Recent Changes (December 4, 2025)

- **Progressive backoff + tweet throttling** - Bot now detects sustained rate limits (3+ consecutive failures) and applies progressive backoff (1h → 4h → 12h) instead of burning through retries. Also added tweet throttling to enforce max 5 tweets/min (~300/hour, ~7200/day) to prevent hitting rate limits in the first place.

## Previous Changes (November 13, 2025)

- **Smart rate limit handling with X-RateLimit-Reset** - Bot now reads Twitter's `X-RateLimit-Reset` header to determine exact retry time instead of blindly waiting 15 minutes. This prevents compounding rate limit issues and respects Twitter's actual rate limit windows. Logs now show the exact reset time (e.g., "retrying after reset at 2025-11-13 12:30:00 UTC").

## Previous Changes (November 12, 2025)

- **Comprehensive NFT lending protocol filtering** - Added exclusion filters for all major NFT lending platforms to prevent loan transactions from being posted as sales:
  - NFTfi (V2, V2.3, V3 - all versions including escrow)
  - Blend (Blur Lending)
  - Arcade (V2 and V3)
  - Gondi (V2 and V3)
  - Filters both loan deposits (collateral) and loan repayments/withdrawals

## Previous Changes (November 9, 2025)

- **Fixed WETH detection** - Bot now correctly detects sales paid in WETH (Wrapped ETH) instead of only native ETH. This fixes OpenSea and marketplace sales being incorrectly skipped.
- **Added Etherscan API fallback** - When Moralis webhooks don't include ERC-20 transfer data, bot automatically checks Etherscan API for WETH transfers. This ensures all marketplace sales are captured regardless of webhook payload contents.

## Previous Changes (November 7, 2025)

- **Added rate limit retry queue** - When Twitter rate limits are hit (HTTP 429), bot now automatically retries after 15 minutes (configurable) instead of failing permanently. Supports up to 10 retry attempts.
- **Fixed token name display** - Single sales now show "Citizen of Vibetown #X has sold..." instead of generic "Token #X has sold..."
- **Added sales-only filtering** - Bot now only posts NFT sales with ETH/WETH payment (≥0.001 ETH), ignoring free transfers, gifts, and airdrops

## Previous Changes (November 6, 2025)

- **Fixed webhook signature verification** - Resolved 401 errors from Moralis webhooks
- **Added Alchemy API fallback** - When Moralis doesn't provide NFT metadata/images, bot automatically fetches from Alchemy's public API
- **Fixed price display for marketplace sales** - Shows "N/A" instead of "$0.00" when exact sale price is unavailable (common with OpenSea/marketplace sales where payment is in separate transaction)
- **Improved logging** - Added visibility into metadata fetching, image download process, and skipped transfers
- Fixed grammar in sweep posts (plural "have sold" vs singular "has sold")
- Fixed duplicate token ID display bug in tweet text
- Added image format conversion (AVIF/WEBP/PNG → JPEG) for Twitter compatibility

## Previous Changes (November 5, 2025)

- Migrated from keyring-based secret management to Replit Secrets
- Configured Flask app to run on 0.0.0.0:5000 for Replit environment
- Set up workflow for automatic bot execution
- Cleaned up requirements.txt and removed keyring dependency
- Added Python .gitignore patterns
- Deployed with Good Vibes Club branding and custom dashboard

## Project Architecture

### Core Components

1. **Flask Web Server** (`nft_sales_bot.py`)
   - Webhook endpoint for Moralis blockchain events
   - Metrics endpoints: `/metrics`, `/metrics/prom`, `/health`, `/ready`
   - Debug endpoints: `/debug/single`, `/debug/sweep`
   - Runs on port 5000 with 0.0.0.0 binding for Replit compatibility

2. **Twitter Integration**
   - Uses Twitter API v1.1 for media uploads
   - Uses Twitter API v2 for posting tweets
   - OAuth 1.0a authentication (Free tier compatible)
   - Supports up to 4 images per tweet

3. **Moralis Webhook Handler**
   - Receives NFT transfer events from Moralis Streams
   - Signature verification for security
   - Tolerant JSON parsing for various event types

4. **Image Processing**
   - Downloads NFT images from IPFS/HTTP sources
   - Creates collages for sweep purchases (up to 12 items per collage)
   - Supports multiple IPFS gateways for reliability
   - 16:9 aspect ratio collages optimized for Twitter

5. **Job Queue System**
   - Background workers (default: 3 workers)
   - Thread pool for parallel image downloads and uploads
   - Idempotency tracking (in-memory or Redis)

## Required Secrets

You must add these secrets in the Replit Secrets panel (Tools → Secrets):

### Twitter/X API Credentials
- `X_APP_KEY` - Twitter API Key
- `X_APP_SECRET` - Twitter API Key Secret
- `X_ACCESS_TOKEN` - Twitter Access Token
- `X_ACCESS_SECRET` - Twitter Access Token Secret

### Moralis Configuration
- `MORALIS_WEBHOOK_SECRET` - Moralis Streams webhook secret for signature verification
- `MORALIS_API_KEY` - (Optional) Moralis API key for REST calls

### How to Get API Keys

**Twitter/X API:**
1. Go to https://developer.twitter.com/
2. Create a new app or use an existing one
3. Generate API keys and access tokens
4. Ensure your app has Read and Write permissions

**Moralis:**
1. Sign up at https://moralis.io/
2. Create a new Stream
3. Copy the webhook secret from the Stream settings
4. Configure the Stream to watch NFT transfers

## Optional Environment Variables

Set these in the Replit Secrets or Environment Variables (not sensitive):

### Collection Monitoring
- `COLLECTION_ADDRESSES` - Comma-separated NFT contract addresses to watch (lowercase)
  - Example: `0xabc123...,0xdef456...`
  - If not set, monitors all collections

### Image & Collage Settings
- `TWITTER_CANVAS_W` - Collage width in pixels (default: 2048)
- `TWITTER_CANVAS_H` - Collage height in pixels (default: 1152)
- `COLLAGE_MAX_ITEMS` - Max items per collage (default: 12)
- `MULTI_COLLAGE_MAX` - Max collages per tweet (default: 4)
- `COLLAGE_GAP_PX` - Gap between images in pixels (default: 24)
- `COLLAGE_BG` - Background color hex code (default: #000000)
- `COLLAGE_JPEG_QUALITY` - JPEG quality 0-100 (default: 90)

### Performance Tuning
- `JOB_WORKERS` - Number of background workers (default: 3)
- `JOB_QUEUE_MAX` - Max jobs in queue (default: 200)
- `UPLOAD_THREADS` - Parallel media uploads (default: 4)
- `DOWNLOAD_THREADS` - Parallel image downloads (default: 6)

### Rate Limit Handling
- `RATE_LIMIT_RETRY_DELAY_MINS` - Minutes to wait before retrying after Twitter rate limit (default: 15)
- `RATE_LIMIT_MAX_RETRIES` - Max retry attempts for rate-limited tweets (default: 10)
- `WEBHOOK_RATE_THRESHOLD` - Webhook arrival rate threshold to trigger throttling (default: 10 webhooks/min)

### Storage & Caching
- `OUTPUT_DIR` - Directory for temporary images (default: ./out)
- `REDIS_URL` - Redis connection string for distributed idempotency (optional)
- `IDEMP_TTL_SECS` - Idempotency cache TTL in seconds (default: 7200)

### Security & Debugging
- `ALLOW_UNSIGNED` - Skip webhook signature verification (default: 0, set to 1 only for testing)
- `IPFS_GATEWAY` - Primary IPFS gateway URL (default: https://ipfs.io/ipfs)

## Usage

### Running the Bot

The bot runs automatically via the configured workflow. To manually start:

```bash
python nft_sales_bot.py
```

To run on a custom port:

```bash
python nft_sales_bot.py --port 8080
```

### Testing Endpoints

**Health Check:**
```bash
curl https://your-repl.replit.dev/health
```

**Metrics (JSON):**
```bash
curl https://your-repl.replit.dev/metrics
```

**Metrics (Prometheus format):**
```bash
curl https://your-repl.replit.dev/metrics/prom
```

**Debug Single Sale:**
```bash
curl -X POST https://your-repl.replit.dev/debug/single
```

**Debug Sweep:**
```bash
curl -X POST https://your-repl.replit.dev/debug/sweep
```

### Configuring Moralis Webhook

1. In your Moralis Stream, set the webhook URL to: `https://your-repl.replit.dev/webhook`
2. Ensure the webhook secret matches what you set in `MORALIS_WEBHOOK_SECRET`
3. Configure the Stream to monitor NFT transfers (ERC-721, ERC-1155)

## File Structure

```
.
├── nft_sales_bot.py      # Main bot application
├── requirements.txt      # Python dependencies
├── README.md            # Original project README
├── replit.md           # This file
├── .gitignore          # Git ignore patterns
└── out/                # Output directory for temporary images (created automatically)
```

## Twitter Free Tier Limits

- **500 posts per month** (17 per 24 hours)
- **15 posts per 15-minute window** (per-window rate limit)
- Bot includes progressive backoff: detects sustained rate limits and waits 1h → 4h → 12h to avoid burning through retries

Upgrade to Basic tier ($200/mo): 10,000 posts per month (100 per 24 hours)

## Dependencies

- **Flask** - Web framework for webhook server
- **requests** - HTTP client library
- **requests-oauthlib** - OAuth 1.0a for Twitter API
- **Pillow** - Image processing and collage creation
- **pycryptodome** - Cryptographic functions for webhook verification
- **redis** - Optional Redis client for distributed caching
- **gunicorn** - Production WSGI server (for deployment)

## Features

### Single Sales
- Posts individual NFT sales with:
  - Token image from IPFS/HTTP
  - OpenSea link
  - Price in ETH (Ξ) and USD
  - Buyer and seller addresses (shortened)

### Sweep Purchases
- Detects multiple NFTs bought by same address in one transaction
- Creates visual collages (up to 4 collages with 12 items each)
- Smart text formatting with token IDs and prices
- Adaptive text truncation to fit Twitter's 280 character limit

### Security
- Webhook signature verification using Keccak-256
- Safe image download with size limits (15MB max)
- Pixel bomb protection
- No secret logging or exposure

### Reliability
- Multi-gateway IPFS fallback
- Idempotency tracking to prevent duplicate posts
- Retry logic for network requests
- Background job queue with error handling

## Troubleshooting

### Bot not posting to Twitter
1. Verify all Twitter API secrets are set correctly
2. Check that your Twitter app has Read and Write permissions
3. Review logs for authentication errors

### Webhook not receiving events
1. Ensure webhook URL is set correctly in Moralis
2. Verify `MORALIS_WEBHOOK_SECRET` matches Moralis Stream settings
3. Check that signature verification is enabled (`ALLOW_UNSIGNED=0`)

### Images not loading
1. Check IPFS gateway connectivity
2. Verify image URLs in the webhook payload
3. Review logs for download errors

### Memory or performance issues
1. Reduce `JOB_WORKERS` if running low on resources
2. Lower `COLLAGE_MAX_ITEMS` for smaller collages
3. Consider adding Redis for better idempotency tracking

## Deployment Notes

This bot is ready to run continuously in Replit. The workflow is configured to:
- Start automatically when the Repl runs
- Listen on port 5000 for webhook events
- Restart automatically on code changes
- Show console output for monitoring

For production deployment outside Replit, consider using `gunicorn` as the WSGI server:
```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port -w 4 nft_sales_bot:app
```

## Support

For issues with:
- **Twitter API**: https://developer.twitter.com/en/support
- **Moralis**: https://moralis.io/support/
- **This bot**: Check logs in the Replit console for error details
