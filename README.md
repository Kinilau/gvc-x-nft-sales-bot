# NFT Sales Bot — Full (keyring-aligned)

- Keyring-only secret handling (same behavior as vibestr_bot.py)
- Single & sweep posting with collages (up to 4 x 12 images)
- Moralis webhook with tolerant parser for Deploy probes
- Metrics endpoints: /metrics, /metrics/prom, /ready, /health
- Debug endpoints: /debug/single, /debug/sweep

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python nft_sales_bot.py          # prompts for secrets on first run and stores in keyring
```

## Reset secrets
```bash
python nft_sales_bot.py --reset-secrets
```

## Configure watched collections
Set env var (comma-separated addresses, lowercase):
```bash
export COLLECTION_ADDRESSES=0xabc...,0xdef...
# PowerShell: $env:COLLECTION_ADDRESSES = "0xabc...,0xdef..."
```
