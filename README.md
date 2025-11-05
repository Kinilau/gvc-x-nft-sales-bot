# NFT Sales Bot

[![Replit](https://img.shields.io/badge/Replit-Ready-blue)](https://replit.com)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)](https://flask.palletsprojects.com)

An automated NFT sales bot that monitors blockchain transactions via Moralis webhooks and posts updates to Twitter/X. The bot handles both single NFT sales and sweep purchases (bulk buys), creating attractive collages for multi-item sales.

## Features

- 🖼️ **Single & Sweep Posting** - Posts individual sales or creates collages (up to 4 x 12 images)
- 🔗 **Moralis Integration** - Webhook receiver with signature verification
- 🐦 **Twitter/X Posting** - OAuth 1.0a (Free tier compatible)
- 📊 **Metrics Endpoints** - `/metrics`, `/metrics/prom`, `/ready`, `/health`
- 🛠️ **Debug Endpoints** - `/debug/single`, `/debug/sweep`
- 🌐 **IPFS Support** - Multi-gateway fallback for reliable image fetching
- ⚡ **Background Workers** - Async job queue with thread pools
- 🔒 **Security** - Webhook signature verification, safe image handling, no secret logging

## Quick Start on Replit

### 1. Add Required Secrets

Go to **Tools → Secrets** and add:

**Twitter/X API Credentials:**
- `X_APP_KEY` - Twitter API Key
- `X_APP_SECRET` - Twitter API Key Secret
- `X_ACCESS_TOKEN` - Twitter Access Token
- `X_ACCESS_SECRET` - Twitter Access Token Secret

**Moralis Configuration:**
- `MORALIS_WEBHOOK_SECRET` - Moralis Streams webhook secret

**Optional:**
- `MORALIS_API_KEY` - Moralis API key (for REST calls)
- `COLLECTION_ADDRESSES` - Comma-separated NFT contract addresses to monitor (e.g., `0xabc...,0xdef...`)

### 2. Get API Keys

**Twitter/X API:**
1. Visit [Twitter Developer Portal](https://developer.twitter.com/)
2. Create a new app or use an existing one
3. Generate API keys and access tokens
4. Ensure your app has **Read and Write** permissions

**Moralis:**
1. Sign up at [Moralis.io](https://moralis.io/)
2. Create a new Stream
3. Copy the webhook secret from Stream settings
4. Configure the Stream to watch NFT transfers (ERC-721, ERC-1155)

### 3. Run the Bot

The bot starts automatically when you run the Repl. Once running:
- The Flask server will listen on port 5000
- Configure your Moralis webhook URL to: `https://your-repl.replit.dev/webhook`

### 4. Deploy to Production

Click the **Deploy** button to publish your bot with:
- 24/7 uptime for continuous webhook monitoring
- **Production-ready Gunicorn WSGI server** (4 workers)
- Automatic scaling and reliability

> **⚠️ Important: Development vs Production**
> 
> - **Development (Run button)**: Uses Flask's built-in development server for testing
> - **Production (Deploy button)**: Uses Gunicorn WSGI server for production workloads
> 
> The deployment is already configured to use Gunicorn. When you click Deploy, your bot will automatically run with:
> ```bash
> gunicorn --bind 0.0.0.0:5000 --reuse-port -w 4 nft_sales_bot:app
> ```
> This provides multi-process concurrency and production-grade reliability for handling webhooks.

## Local Development

If you want to run locally (outside Replit):

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export X_APP_KEY="your_key"
export X_APP_SECRET="your_secret"
export X_ACCESS_TOKEN="your_token"
export X_ACCESS_SECRET="your_token_secret"
export MORALIS_WEBHOOK_SECRET="your_webhook_secret"

# Optional: Configure collections to monitor
export COLLECTION_ADDRESSES="0xabc...,0xdef..."

# Run the bot
python nft_sales_bot.py
```

## Configuration

### Environment Variables

Set these in Replit Secrets or as environment variables:

#### Collection Monitoring
- `COLLECTION_ADDRESSES` - Comma-separated NFT contract addresses (lowercase). If not set, monitors all collections.

#### Image & Collage Settings
- `TWITTER_CANVAS_W` - Collage width in pixels (default: 2048)
- `TWITTER_CANVAS_H` - Collage height in pixels (default: 1152)
- `COLLAGE_MAX_ITEMS` - Max items per collage (default: 12)
- `MULTI_COLLAGE_MAX` - Max collages per tweet (default: 4)
- `COLLAGE_GAP_PX` - Gap between images in pixels (default: 24)
- `COLLAGE_BG` - Background color hex code (default: #000000)
- `COLLAGE_JPEG_QUALITY` - JPEG quality 0-100 (default: 90)

#### Performance Tuning
- `JOB_WORKERS` - Number of background workers (default: 3)
- `JOB_QUEUE_MAX` - Max jobs in queue (default: 200)
- `UPLOAD_THREADS` - Parallel media uploads (default: 4)
- `DOWNLOAD_THREADS` - Parallel image downloads (default: 6)

#### Storage & Caching
- `OUTPUT_DIR` - Directory for temporary images (default: ./out)
- `REDIS_URL` - Redis connection string for distributed idempotency (optional)
- `IDEMP_TTL_SECS` - Idempotency cache TTL in seconds (default: 7200)

#### Security & Debugging
- `ALLOW_UNSIGNED` - Skip webhook signature verification (default: 0, set to 1 only for testing)
- `IPFS_GATEWAY` - Primary IPFS gateway URL (default: https://ipfs.io/ipfs)

## API Endpoints

### Health & Metrics

- **GET /health** - Health check endpoint
- **GET /ready** - Readiness check
- **GET /metrics** - JSON metrics
- **GET /metrics/prom** - Prometheus-format metrics

### Webhook

- **POST /webhook** - Moralis Streams webhook endpoint (requires signature verification)

### Debug

- **POST /debug/single** - Test single NFT sale post
- **POST /debug/sweep** - Test sweep purchase post with collage

## Testing

```bash
# Health check
curl https://your-repl.replit.dev/health

# View metrics
curl https://your-repl.replit.dev/metrics

# Test single sale (debug)
curl -X POST https://your-repl.replit.dev/debug/single

# Test sweep (debug)
curl -X POST https://your-repl.replit.dev/debug/sweep
```

## How It Works

### Single Sales
Posts individual NFT sales with:
- Token image from IPFS/HTTP
- OpenSea link
- Price in ETH (Ξ) and USD
- Buyer and seller addresses (shortened)

### Sweep Purchases
Detects multiple NFTs bought by same address in one transaction:
- Creates visual collages (up to 4 collages with 12 items each)
- Smart text formatting with token IDs and prices
- Adaptive text truncation to fit Twitter's 280 character limit

## Architecture

- **Flask Web Server** - Webhook receiver and API endpoints
- **Background Workers** - Async job queue (default: 3 workers)
- **Thread Pools** - Parallel image downloads and uploads
- **Idempotency Store** - In-memory or Redis-based deduplication
- **IPFS Multi-Gateway** - Fallback support for reliable image fetching

## Troubleshooting

### Bot not posting to Twitter
1. Verify all Twitter API secrets are set correctly in Replit Secrets
2. Check that your Twitter app has **Read and Write** permissions
3. Review console logs for authentication errors

### Webhook not receiving events
1. Ensure webhook URL is set correctly in Moralis Stream
2. Verify `MORALIS_WEBHOOK_SECRET` matches Moralis settings
3. Check that signature verification is enabled (`ALLOW_UNSIGNED=0`)

### Images not loading
1. Check IPFS gateway connectivity
2. Verify image URLs in the webhook payload
3. Review logs for download errors

### Memory or performance issues
1. Reduce `JOB_WORKERS` if running low on resources
2. Lower `COLLAGE_MAX_ITEMS` for smaller collages
3. Consider adding Redis for better idempotency tracking

## Dependencies

- **Flask** >= 3.0 - Web framework
- **requests** >= 2.31.0 - HTTP client
- **requests-oauthlib** >= 1.3.1 - OAuth 1.0a for Twitter
- **Pillow** >= 10.0.0 - Image processing
- **pycryptodome** >= 3.19.0 - Cryptographic functions
- **redis** >= 5.0.0 - Optional Redis client
- **gunicorn** >= 21.2.0 - Production WSGI server

## License

MIT

## Support

- **Twitter API**: [Developer Support](https://developer.twitter.com/en/support)
- **Moralis**: [Support Center](https://moralis.io/support/)
- **Issues**: Check console logs in Replit for error details

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
