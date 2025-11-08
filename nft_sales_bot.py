#!/usr/bin/env python3
# nft_sales_bot.py — Full bot with vibestr-style keyring-only secret handling
# - Single sales: raw token image + OpenSea URL, ETH symbol (Ξ) + $USD
# - Sweeps: up to 4 collages (12 images each), no links in text, breakdown lines prioritize token+price then seller if room
# - Twitter posting: v1.1 media/upload + v2 /tweets (OAuth 1.0a user context) — Free-plan compatible
# - Webhook: Moralis Streams signature verification; tolerant JSON parsing to pass "Deploy" probe
# - Async: background job queue; threaded downloads/collages/uploads
# - Safety: size/time limits, IPFS multi-gateway, idempotency, safe logging
# - Metrics: /metrics (JSON), /metrics/prom (Prometheus), /health, /ready
# - Debug: /debug/single, /debug/sweep

import os, sys, json, time, random, threading, queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry
from flask import Flask, request, Response
from requests_oauthlib import OAuth1
from PIL import Image, ImageColor, UnidentifiedImageError

# Custom exceptions
class RateLimitError(Exception):
    """Raised when Twitter API rate limit is hit (HTTP 429)"""
    pass

# ================== Replit Secrets (environment variables) ==================
import argparse

SECRET_KEYS_REQUIRED = [
    "X_APP_KEY",
    "X_APP_SECRET",
    "X_ACCESS_TOKEN",
    "X_ACCESS_SECRET",
    "MORALIS_WEBHOOK_SECRET",
]
SECRET_KEYS_OPTIONAL = [
    "MORALIS_API_KEY",
]

def get_secret(name: str, optional: bool = False) -> Optional[str]:
    val = os.environ.get(name)
    if val is None and not optional:
        print(f"[WARNING] Required secret '{name}' is not set in environment variables.", flush=True)
        print(f"[INFO] Please add '{name}' to Replit Secrets.", flush=True)
    return val

# CLI flags
_argparser = argparse.ArgumentParser(add_help=True)
_argparser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
_args, _unknown = _argparser.parse_known_args()

# Load secrets from environment variables (Replit Secrets)
X_APP_KEY       = get_secret("X_APP_KEY")
X_APP_SECRET    = get_secret("X_APP_SECRET")
X_ACCESS_TOKEN  = get_secret("X_ACCESS_TOKEN")
X_ACCESS_SECRET = get_secret("X_ACCESS_SECRET")
MORALIS_WEBHOOK_SECRET = get_secret("MORALIS_WEBHOOK_SECRET")
MORALIS_API_KEY = get_secret("MORALIS_API_KEY", optional=True)

PORT = int(_args.port)

# -------------------------
# Environment configuration (non-secret config still via ENV)
# -------------------------
ALLOW_UNSIGNED          = os.environ.get("ALLOW_UNSIGNED", "0").strip().lower() in ("1","true","yes","on")
COLLECTIONS             = {a.strip().lower() for a in os.environ.get("COLLECTION_ADDRESSES","").split(",") if a.strip()}
IPFS_GATEWAY            = os.environ.get("IPFS_GATEWAY", "https://ipfs.io/ipfs").rstrip("/")

# Sweep collage settings
CANVAS_W = int(os.environ.get("TWITTER_CANVAS_W", "2048"))
CANVAS_H = int(os.environ.get("TWITTER_CANVAS_H", "1152"))
COLLAGE_MAX_ITEMS = int(os.environ.get("COLLAGE_MAX_ITEMS", "12"))  # max tiles per collage
MULTI_COLLAGE_MAX = int(os.environ.get("MULTI_COLLAGE_MAX", "4"))   # max collages per tweet
COLLAGE_GAP_PX = int(os.environ.get("COLLAGE_GAP_PX", "24"))
COLLAGE_BG = os.environ.get("COLLAGE_BG", "#000000")
COLLAGE_JPEG_QUALITY = int(os.environ.get("COLLAGE_JPEG_QUALITY", "90"))

# Async workers
JOB_WORKERS = int(os.environ.get("JOB_WORKERS", "3"))
JOB_QUEUE_MAX = int(os.environ.get("JOB_QUEUE_MAX", "200"))

# Rate limit retry settings
RATE_LIMIT_RETRY_DELAY_MINS = int(os.environ.get("RATE_LIMIT_RETRY_DELAY_MINS", "15"))
RATE_LIMIT_MAX_RETRIES = int(os.environ.get("RATE_LIMIT_MAX_RETRIES", "10"))

# Parallelism inside a job
UPLOAD_THREADS = int(os.environ.get("UPLOAD_THREADS", "4"))     # ≤ 4 media per tweet
DOWNLOAD_THREADS = int(os.environ.get("DOWNLOAD_THREADS", "6")) # per-collage image fetch

# Output directory
OUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./out")).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Idempotency (in-memory TTL seconds). Optional Redis if REDIS_URL set.
IDEMP_TTL_SECS = int(os.environ.get("IDEMP_TTL_SECS", str(2 * 60 * 60)))  # 2h
REDIS_URL = os.environ.get("REDIS_URL", "").strip()

# Flask hardening
MAX_WEBHOOK_BYTES = int(os.environ.get("MAX_WEBHOOK_BYTES", str(2 * 1024 * 1024)))  # 2MB

# Requests tuning
HTTP_TIMEOUT = (5, 30)  # connect, read
RETRY_POLICY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET", "POST"),
    raise_on_status=False,
)

# Image safety
ETH_SYM = "Ξ"
COMMON_IPFS = [
    "https://nftstorage.link/ipfs",
    "https://ipfs.io/ipfs",
    "https://cloudflare-ipfs.com/ipfs",
    "https://gateway.pinata.cloud/ipfs",
    "https://dweb.link/ipfs",
]
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))  # 15MB
Image.MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(120_000_000)))

# HTTP session
S = requests.Session()
S.headers.update({"User-Agent": "nft-sales-bot/1.0"})
S.mount("https://", HTTPAdapter(max_retries=RETRY_POLICY))
S.mount("http://",  HTTPAdapter(max_retries=RETRY_POLICY))

# Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_WEBHOOK_BYTES
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

def log(msg: str, lvl="INFO"):
    if isinstance(msg, str):
        for k in ("MORALIS_WEBHOOK_SECRET","X_APP_KEY","X_APP_SECRET","X_ACCESS_TOKEN","X_ACCESS_SECRET"):
            if k in msg: msg = "[redacted]"
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] [{lvl}] {msg}", flush=True)

# -------------------------
# Metrics (thread-safe)
# -------------------------
_METRICS_LOCK = threading.Lock()
_METRICS = {
    "start_ts": time.time(),
    "jobs_enqueued_total": 0,
    "jobs_processed_total": {"single": 0, "sweep": 0},
    "jobs_failed_total": {"single": 0, "sweep": 0},
    "jobs_retried_total": {"single": 0, "sweep": 0},
    "tweets_posted_total": 0,
    "media_uploads_total": 0,
    "media_uploads_failed_total": 0,
}
def _m_inc(path: tuple, n: int = 1):
    with _METRICS_LOCK:
        ref = _METRICS
        for key in path[:-1]:
            ref = ref[key]
        ref[path[-1]] = ref.get(path[-1], 0) + n

# -------------------------
# Idempotency store
# -------------------------
class IdempotencyStore:
    def __init__(self, ttl_secs: int):
        self.ttl = ttl_secs
        self.mem: Dict[str, float] = {}
        self.redis = None
        if REDIS_URL:
            try:
                import redis  # type: ignore
                self.redis = redis.from_url(REDIS_URL, decode_responses=True)
                log("Idempotency: Redis backend enabled", "INFO")
            except Exception as e:
                log(f"Redis unavailable, falling back to in-memory: {e}", "WARN")

    def key(self, tx: str, buyer: str, token_addr: str) -> str:
        return f"{tx.lower()}|{buyer.lower()}|{token_addr.lower()}"

    def seen(self, tx: str, buyer: str, token_addr: str) -> bool:
        k = self.key(tx, buyer, token_addr)
        now = time.time()
        if self.redis:
            try:
                return self.redis.exists(k) == 1
            except Exception as e:
                log(f"Redis exists error: {e}", "WARN")
        # prune occasionally
        if self.mem and int(now) % 97 == 0:
            expired = [kk for kk, ts in self.mem.items() if ts + self.ttl < now]
            for kk in expired:
                self.mem.pop(kk, None)
        return (k in self.mem) and (self.mem[k] + self.ttl > now)

    def mark(self, tx: str, buyer: str, token_addr: str) -> None:
        k = self.key(tx, buyer, token_addr)
        now = time.time()
        if self.redis:
            try:
                self.redis.set(k, "1", nx=True, ex=self.ttl)
                return
            except Exception as e:
                log(f"Redis set error: {e}", "WARN")
        self.mem[k] = now

IDEMP = IdempotencyStore(IDEMP_TTL_SECS)

# -------------------------
# Formatting helpers
# -------------------------
def fmt_eth(x: Optional[float]) -> str:
    if x is None or x <= 0.0001: 
        return f"N/A {ETH_SYM}"
    return f"{x:0.2f} {ETH_SYM}"

def fmt_usd(x: Optional[float]) -> str:
    if x is None or x <= 0.01: return "$N/A"
    v = float(x)
    if v >= 1_000_000_000: return f"${v/1_000_000_000:0.2f}B"
    if v >= 1_000_000:     return f"${v/1_000_000:0.2f}M"
    if v >= 1_000:         return f"${v/1_000:0.2f}K"
    return f"${v:0.2f}"

def fmt_usd_compact(x: Optional[float]) -> str:
    if x is None: return "$N/A"
    v = float(x)
    if v >= 1_000_000_000: return f"${v/1_000_000_000:0.1f}B"
    if v >= 1_000_000:     return f"${v/1_000_000:0.1f}M"
    if v >= 1_000:         return f"${v/1_000:0.1f}K"
    return f"${v:0.2f}"

def shorten_addr(addr: str) -> str:
    if not addr or len(addr) < 10: 
        return addr or ""
    a = addr.lower()
    if not a.startswith("0x"): 
        return a
    return a[:6] + "…" + a[-4:]

def tweet_fits(txt: str, limit: int = 280) -> bool:
    return len(txt) <= limit

def opensea_asset_url(contract: str, token_id: str) -> str:
    return f"https://opensea.io/assets/ethereum/{contract}/{token_id}"

# -------------------------
# Price helpers
# -------------------------
def get_eth_usd() -> Optional[float]:
    try:
        r = S.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids":"ethereum","vs_currencies":"usd"},
            timeout=HTTP_TIMEOUT
        )
        if r.status_code != 200:
            return None
        p = (r.json() or {}).get("ethereum",{}).get("usd")
        return float(p) if p is not None else None
    except Exception as e:
        log(f"USD price fetch failed: {e}", "WARN")
        return None

# -------------------------
# IPFS / image helpers
# -------------------------
def _strip_ipfs_prefix(s: str) -> str:
    if s.startswith("ipfs://"): s = s[7:]
    if s.startswith("ipfs/"): s = s[5:]
    return s.lstrip("/")

def ipfs_candidates(url_or_ipfs: str) -> List[str]:
    if not url_or_ipfs:
        return []
    gateways = [IPFS_GATEWAY] + [g for g in COMMON_IPFS if g.rstrip("/") != IPFS_GATEWAY.rstrip("/")]
    if url_or_ipfs.startswith("http"):
        pu = urlparse(url_or_ipfs)
        path = pu.path or ""
        try:
            idx = path.lower().index("/ipfs/")
            tail = path[idx+len("/ipfs/"):]
            ipfs_path = _strip_ipfs_prefix(tail)
        except ValueError:
            return [url_or_ipfs]
        return [f"{g}/{ipfs_path}" for g in gateways]
    else:
        ipfs_path = _strip_ipfs_prefix(url_or_ipfs)
        return [f"{g}/{ipfs_path}" for g in gateways]

def _download_limited(u: str) -> Optional[bytes]:
    try:
        with S.get(u, timeout=HTTP_TIMEOUT, stream=True) as r:
            if r.status_code != 200:
                return None
            total = 0
            chunks = []
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    log(f"image too large (> {MAX_IMAGE_BYTES} bytes): {u}", "WARN")
                    return None
                chunks.append(chunk)
            return b"".join(chunks)
    except Exception as e:
        log(f"download failed {u}: {e}", "WARN")
        return None

def download_image_any(url_or_ipfs: str) -> Optional[Path]:
    for cand in ipfs_candidates(url_or_ipfs):
        b = _download_limited(cand)
        if b:
            try:
                temp_p = OUT_DIR / f"nft_{int(time.time()*1000)}_temp.img"
                temp_p.write_bytes(b)
                
                with Image.open(str(temp_p)) as img:
                    img_rgb = img.convert("RGB")
                    output_p = OUT_DIR / f"nft_{int(time.time()*1000)}.jpg"
                    img_rgb.save(str(output_p), "JPEG", quality=95, optimize=True)
                
                try: temp_p.unlink(missing_ok=True)
                except: pass
                
                return output_p
            except (UnidentifiedImageError, OSError) as e:
                log(f"image conversion failed for {cand}: {e}", "WARN")
                try: temp_p.unlink(missing_ok=True)
                except: pass
                continue
    return None

# -------------------------
# Collage (sweeps)
# -------------------------
def _layout_boxes(n: int, W: int, H: int, gap: int) -> List[Tuple[int,int,int,int]]:
    """
    Returns [(x,y,w,h), ...] for n tiles within a 16:9 canvas.
    Stable grids up to 12 items; centers partial last rows.
    """
    n = max(1, min(n, COLLAGE_MAX_ITEMS))
    presets = {
        1:(1,1),
        2:(2,1), 3:(3,1),
        4:(2,2),
        5:(3,2),
        6:(3,2),
        7:(4,2),
        8:(4,2),
        9:(3,3),
        10:(5,2),
        11:(4,3),
        12:(4,3),
    }
    cols, rows = presets.get(n, (4,3))
    cw = (W - gap*(cols+1)) // cols
    ch = (H - gap*(rows+1)) // rows

    boxes: List[Tuple[int,int,int,int]] = []
    full_rows, rem = divmod(n, cols)

    for r in range(full_rows):
        y = gap + r*(ch+gap)
        total_w = cols*cw + (cols-1)*gap
        x = (W - total_w)//2
        for c in range(cols):
            boxes.append((x + c*(cw+gap), y, cw, ch))

    if rem:
        y = gap + full_rows*(ch+gap)
        total_w = rem*cw + (rem-1)*gap
        x = (W - total_w)//2
        for c in range(rem):
            boxes.append((x + c*(cw+gap), y, cw, ch))

    return boxes[:n]

def _paste_fit(img: Image.Image, box, canvas: Image.Image):
    x, y, w, h = box
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (0,0,0,0))
    bg.paste(img, (0,0), img)
    img = bg.convert("RGB")
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return
    scale = min(w/iw, h/ih)
    nw, nh = max(1, int(iw*scale)), max(1, int(ih*scale))
    imr = img.resize((nw, nh), Image.LANCZOS)
    px = x + (w - nw)//2
    py = y + (h - nh)//2
    canvas.paste(imr, (px, py))

def build_collage(image_paths: List[Path], out_path: Path) -> Path:
    n = max(0, min(len(image_paths), COLLAGE_MAX_ITEMS))
    if n == 0:
        raise RuntimeError("No images for collage")
    bg = ImageColor.getrgb(COLLAGE_BG)
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), bg)
    boxes = _layout_boxes(n, CANVAS_W, CANVAS_H, COLLAGE_GAP_PX)
    for i, p in enumerate(image_paths[:n]):
        try:
            with Image.open(str(p)) as im:
                _paste_fit(im, boxes[i], canvas)
        except (UnidentifiedImageError, OSError) as e:
            log(f"collage skip {p}: {e}", "WARN")
    canvas.save(str(out_path), "JPEG", quality=COLLAGE_JPEG_QUALITY, optimize=True, progressive=True)
    return out_path

# -------------------------
# Tweet text builders
# -------------------------
def single_sale_text(token_name: str, token_id: str, price_eth: Optional[float],
                     price_usd: Optional[float], buyer: str, seller: str, url: str) -> str:
    return (
        f"{token_name} {token_id} has sold for {fmt_eth(price_eth)} ({fmt_usd(price_usd)})\n\n"
        f"Buyer: {shorten_addr(buyer)}\n"
        f"Seller: {shorten_addr(seller)}\n\n"
        f"URL: {url}"
    ).strip()

def build_breakdown_lines(items: List[Dict[str, Any]], with_seller: bool) -> List[str]:
    lines = []
    for it in items:
        tid = it.get("token_id")
        pe  = it.get("price_eth")
        line = f"Token #{tid} sold for {fmt_eth(pe)}"
        if with_seller and it.get("from"):
            line += f" {shorten_addr(it['from'])}"
        lines.append(line)
    return lines

def sweep_text_prioritized(items: List[Dict[str, Any]], total_eth: Optional[float], total_usd: Optional[float],
                           buyer: str) -> str:
    count = len(items)
    verb = "has" if count == 1 else "have"
    header_full   = f"{count} Citizens of Vibetown {verb} sold for {fmt_eth(total_eth)} ({fmt_usd(total_usd)})"
    header_comp   = f"{count} Citizens of Vibetown {verb} sold for {fmt_eth(total_eth)} ({fmt_usd_compact(total_usd)})"
    buyer_line    = f"Buyer: {shorten_addr(buyer)}"

    def assemble(header_line: str, breakdown: List[str]) -> str:
        return "\n".join([header_line, ""] + breakdown + ["", buyer_line]).strip()

    # 1) with sellers
    bd = build_breakdown_lines(items, with_seller=True)
    txt = assemble(header_full, bd)
    if tweet_fits(txt): return txt
    # 2) without sellers
    bd = build_breakdown_lines(items, with_seller=False)
    txt = assemble(header_full, bd)
    if tweet_fits(txt): return txt

    # 3) truncate (keep USD)
    def assemble_with_trunc(header_line: str) -> str:
        if not bd: return "\n".join([header_line, "", buyer_line]).strip()
        lo, hi, best = 1, len(bd), 1
        while lo <= hi:
            mid = (lo + hi) // 2
            kept = bd[:mid]
            more = len(bd) - mid
            candidate = "\n".join([header_line, ""] + kept + ([f"+{more} more"] if more>0 else []) + ["", buyer_line]).strip()
            if tweet_fits(candidate):
                best = mid; lo = mid + 1
            else:
                hi = mid - 1
        kept = bd[:best]; more = len(bd) - best
        return "\n".join([header_line, ""] + kept + ([f"+{more} more"] if more>0 else []) + ["", buyer_line]).strip()

    txt = assemble_with_trunc(header_full)
    if tweet_fits(txt): return txt

    # 4) compact USD + truncation
    txt = assemble_with_trunc(header_comp)
    if tweet_fits(txt): return txt

    # 5) last resort
    return "\n".join([header_comp, "", buyer_line]).strip()

# -------------------------
# Twitter (Free-plan friendly)
# -------------------------
def oauth1() -> Optional[OAuth1]:
    if not all([X_APP_KEY, X_APP_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET]):
        log("Twitter creds missing; cannot post.", "ERROR")
        return None
    return OAuth1(X_APP_KEY, X_APP_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET)

def upload_media_v11(auth: OAuth1, image_path: str) -> Optional[str]:
    try:
        url = "https://upload.twitter.com/1.1/media/upload.json"
        with open(image_path, "rb") as f:
            files = {"media": (Path(image_path).name, f, "application/octet-stream")}
            r = S.post(url, auth=auth, files=files, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            _m_inc(("media_uploads_failed_total",))
            log(f"media/upload failed [{r.status_code}]: {r.text}", "ERROR")
            return None
        _m_inc(("media_uploads_total",))
        return (r.json() or {}).get("media_id_string")
    except Exception as e:
        _m_inc(("media_uploads_failed_total",))
        log(f"media upload error: {e}", "ERROR")
        return None

def post_tweet_v2(auth: OAuth1, text: str, media_ids: Optional[List[str]]) -> bool:
    url = "https://api.twitter.com/2/tweets"
    payload: Dict[str, Any] = {"text": text}
    if media_ids:
        payload["media"] = {"media_ids": media_ids}
    r = S.post(url, auth=auth, json=payload, timeout=HTTP_TIMEOUT)
    if r.status_code == 429:
        log(f"/2/tweets rate limited [429]: {r.text}", "WARN")
        raise RateLimitError(f"Twitter API rate limit hit: {r.text}")
    if r.status_code not in (200, 201):
        log(f"/2/tweets failed [{r.status_code}]: {r.text}", "ERROR")
        return False
    _m_inc(("tweets_posted_total",))
    log(f"Tweet posted: {(r.json() or {}).get('data',{}).get('id')}", "INFO")
    return True

# -------------------------
# Moralis parsing & metadata
# -------------------------
def verify_moralis_sig(raw: bytes, header_sig: str) -> bool:
    if ALLOW_UNSIGNED:
        return True
    if not MORALIS_WEBHOOK_SECRET or not header_sig:
        return False
    sig = header_sig.strip().lower()
    if sig.startswith("0x"): sig = sig[2:]
    try:
        from Crypto.Hash import keccak  # pycryptodome
        k = keccak.new(digest_bits=256)
        k.update(raw + MORALIS_WEBHOOK_SECRET.encode("utf-8"))
        calc = k.hexdigest()
        return (len(sig) == len(calc)) and all(a == b for a, b in zip(sig, calc))
    except Exception as e:
        log(f"Signature verify error: {e}", "ERROR")
        return False

def is_watched_contract(addr: str) -> bool:
    if not COLLECTIONS:
        return True
    return addr.lower() in COLLECTIONS

def parse_nft_transfers(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    transfers: List[Dict[str, Any]] = []
    for k in ("nftTransfers","erc721_transfers","nft_transfers","erc1155_transfers"):
        for t in (payload.get(k) or []):
            transfers.append({
                "token_address": (t.get("token_address") or t.get("contract") or "").lower(),
                "token_id": str(t.get("token_id") or t.get("tokenId") or ""),
                "from": (t.get("from_address") or t.get("from") or t.get("fromAddress") or "").lower(),
                "to": (t.get("to_address") or t.get("to") or t.get("toAddress") or "").lower(),
                "tx": (t.get("transaction_hash") or t.get("tx_hash") or t.get("transactionHash") or ""),
                "value": t.get("value")
            })
    transfers = [t for t in transfers if t["token_address"] and t["token_id"] and is_watched_contract(t["token_address"])]
    return transfers

def estimate_tx_total_eth(payload: Dict[str, Any], tx_hash: str) -> Optional[float]:
    vals = []
    for k in ("txs", "transactions", "logs", "native_transactions"):
        for x in (payload.get(k) or []):
            h = (x.get("hash") or x.get("transaction_hash") or "").lower()
            if h and h == tx_hash.lower():
                v = x.get("value")
                try:
                    if isinstance(v, str):
                        vals.append(int(v))
                    elif isinstance(v, (int, float)):
                        vals.append(int(v))
                except Exception:
                    continue
    if vals:
        wei = sum(vals)
        return wei / 1e18
    return None

def get_token_metadata_from_payload(contract: str, token_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    metas = payload.get("nft_metadata") or payload.get("nfts") or []
    for m in metas:
        c = (m.get("token_address") or m.get("contract") or "").lower()
        tid = str(m.get("token_id") or m.get("tokenId") or "")
        if c == contract.lower() and tid == token_id:
            return m
    return {}

def resolve_image_url_from_meta(meta: Dict[str, Any]) -> Optional[str]:
    if not meta: return None
    nm = (meta.get("normalized_metadata") or meta.get("metadata") or {})
    cand = nm.get("image") or nm.get("image_url")
    if not cand and meta.get("image"):
        if isinstance(meta["image"], dict):
            cand = meta["image"].get("originalUrl") or meta["image"].get("pngUrl") or meta["image"].get("url")
        elif isinstance(meta["image"], str):
            cand = meta["image"]
    return cand

def token_display_name(meta: Dict[str, Any], default_collection: str) -> str:
    nm = (meta.get("name") or "").strip()
    coll = (meta.get("collectionTitle") or meta.get("collection_name") or meta.get("collection") or "").strip()
    if nm: return nm
    if coll: return coll
    
    if default_collection.lower() == "0xb8ea78fcacef50d41375e44e6814ebba36bb33c4":
        return "Citizen of Vibetown"
    return "Token"

def fetch_metadata_from_alchemy(contract: str, token_id: str) -> Dict[str, Any]:
    """Fallback: fetch NFT metadata from Alchemy's public API when Moralis doesn't provide it."""
    try:
        url = "https://eth-mainnet.g.alchemy.com/nft/v3/demo/getNFTMetadata"
        params = {
            "contractAddress": contract,
            "tokenId": token_id,
            "refreshCache": "false"
        }
        r = S.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            img_data = data.get("image", {})
            image_url = (img_data.get("cachedUrl") or 
                        img_data.get("originalUrl") or 
                        img_data.get("thumbnailUrl"))
            
            return {
                "token_address": contract,
                "token_id": token_id,
                "name": data.get("name", ""),
                "normalized_metadata": {
                    "image": image_url
                }
            }
    except Exception as e:
        log(f"Alchemy API fallback failed for {contract} #{token_id}: {e}", "WARN")
    return {}

# -------------------------
# Posting flows
# -------------------------
def image_path_for_token(contract: str, token_id: str, payload: Dict[str, Any]) -> Optional[Path]:
    meta = get_token_metadata_from_payload(contract, token_id, payload)
    img = resolve_image_url_from_meta(meta)
    
    if not img:
        log(f"No image in Moralis metadata for {contract} #{token_id}, trying Alchemy fallback", "INFO")
        meta = fetch_metadata_from_alchemy(contract, token_id)
        img = resolve_image_url_from_meta(meta)
        
    if not img:
        log(f"No image URL found for {contract} #{token_id} after fallback", "WARN")
        return None
    
    return download_image_any(img)

def post_single_sale(contract: str, token_id: str, buyer: str, seller: str,
                     price_eth: Optional[float], payload: Dict[str, Any]) -> None:
    eth_usd = get_eth_usd()
    price_usd = (price_eth * eth_usd) if (price_eth is not None and eth_usd is not None) else None

    meta = get_token_metadata_from_payload(contract, token_id, payload)
    token_name = token_display_name(meta, contract)
    url = opensea_asset_url(contract, token_id)
    
    if f"#{token_id}" in token_name:
        text = single_sale_text(token_name, "", price_eth, price_usd, buyer, seller, url)
    else:
        text = single_sale_text(token_name, f"#{token_id}", price_eth, price_usd, buyer, seller, url)

    img_path = image_path_for_token(contract, token_id, payload)
    auth = oauth1()
    if not auth:
        return
    media_ids = None
    if img_path and Path(img_path).exists():
        mid = upload_media_v11(auth, str(img_path))
        media_ids = [mid] if mid else None
        try: Path(img_path).unlink(missing_ok=True)
        except: pass
    else:
        log(f"single: no image for {contract} #{token_id}; posting text only", "WARN")
    post_tweet_v2(auth, text, media_ids)

def _download_one_image(it: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Path]:
    try:
        return image_path_for_token(it["token_address"], it["token_id"], payload)
    except Exception as e:
        log(f"download image error for {it.get('token_id')}: {e}", "WARN")
        return None

def _build_and_upload_collage(chunk: List[Dict[str, Any]], payload: Dict[str, Any], auth: OAuth1,
                              jpeg_quality: int) -> Optional[str]:
    paths: List[Path] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        futures = [pool.submit(_download_one_image, it, payload) for it in chunk]
        for fut in as_completed(futures):
            p = fut.result()
            if p: paths.append(p)
    if not paths:
        return None

    out = OUT_DIR / f"{int(time.time()*1000)}_collage.jpg"
    global COLLAGE_JPEG_QUALITY
    old_q = COLLAGE_JPEG_QUALITY
    try:
        COLLAGE_JPEG_QUALITY = jpeg_quality
        build_collage(paths, out)
        mid = upload_media_v11(auth, str(out))
        return mid
    finally:
        COLLAGE_JPEG_QUALITY = old_q
        try: out.unlink(missing_ok=True)
        except: pass
        for p in paths:
            try: Path(p).unlink(missing_ok=True)
            except: pass

def post_sweep(buyer: str, token_address: str, items: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    txh = items[0]["tx"]
    per: List[Optional[float]] = []
    have_all = True
    for it in items:
        v = it.get("value")
        pe = None
        if v is not None:
            try: pe = int(v)/1e18
            except: pe = None
        else:
            have_all = False
        per.append(pe)

    total_eth = sum([p for p in per if p is not None]) if have_all else estimate_tx_total_eth(payload, txh)
    if total_eth is None and any(per):
        total_eth = sum([p for p in per if p is not None])

    if not have_all and total_eth is not None and len(items) > 0:
        per = [total_eth/len(items) for _ in items]

    eth_usd = get_eth_usd()
    total_usd = (total_eth * eth_usd) if (total_eth is not None and eth_usd is not None) else None

    for i, it in enumerate(items):
        if i < len(per):
            it["price_eth"] = per[i]

    text = sweep_text_prioritized(items, total_eth, total_usd, buyer)

    auth = oauth1()
    if not auth:
        return

    media_ids: List[str] = []
    BATCH_SIZE = min(COLLAGE_MAX_ITEMS, 12)
    chunks = [items[i:i+BATCH_SIZE] for i in range(0, len(items), BATCH_SIZE)]
    chunks = chunks[:MULTI_COLLAGE_MAX]

    def quality_for(n: int) -> int:
        if n >= 10: return min(COLLAGE_JPEG_QUALITY, 85)
        if n >= 8:  return min(COLLAGE_JPEG_QUALITY, 88)
        return COLLAGE_JPEG_QUALITY

    workers = max(1, min(len(chunks), UPLOAD_THREADS))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_build_and_upload_collage, chunk, payload, auth, quality_for(len(chunk)))
                   for chunk in chunks]
        for fut in as_completed(futures):
            mid = fut.result()
            if mid:
                media_ids.append(mid)

    post_tweet_v2(auth, text, media_ids if media_ids else None)

# -------------------------
# Async job queue
# -------------------------
@dataclass
class PostJob:
    kind: str   # "single" | "sweep"
    data: dict
    attempts: int = 0
    created_ts: float = field(default_factory=time.time)

JOB_Q: "queue.Queue[PostJob]" = queue.Queue(maxsize=JOB_QUEUE_MAX)
_WORKERS: List[threading.Thread] = []

def _push_job(job: PostJob) -> bool:
    try:
        JOB_Q.put_nowait(job)
        _m_inc(("jobs_enqueued_total",))
        return True
    except queue.Full:
        log("job queue full; dropping job", "WARN")
        return False

def _job_backoff(attempt: int) -> float:
    base = min(4.0, 0.5 * (2 ** max(0, attempt)))
    return base + random.uniform(0, 0.3)

def _run_job(job: PostJob):
    try:
        if job.kind == "single":
            post_single_sale(**job.data)
            _m_inc(("jobs_processed_total","single"))
        elif job.kind == "sweep":
            post_sweep(**job.data)
            _m_inc(("jobs_processed_total","sweep"))
        else:
            log(f"unknown job kind: {job.kind}", "WARN")
    except RateLimitError as e:
        job.attempts += 1
        _m_inc(("jobs_retried_total", job.kind if job.kind in ("single","sweep") else "single"))
        if job.attempts < RATE_LIMIT_MAX_RETRIES:
            delay_secs = RATE_LIMIT_RETRY_DELAY_MINS * 60
            log(f"Rate limited ({job.kind}), retrying in {RATE_LIMIT_RETRY_DELAY_MINS} minutes (attempt {job.attempts}/{RATE_LIMIT_MAX_RETRIES})", "WARN")
            time.sleep(delay_secs)
            _push_job(job)
        else:
            _m_inc(("jobs_failed_total", job.kind if job.kind in ("single","sweep") else "single"))
            log(f"Job permanently failed after {job.attempts} rate limit retries: {e}", "ERROR")
    except Exception as e:
        job.attempts += 1
        _m_inc(("jobs_retried_total", job.kind if job.kind in ("single","sweep") else "single"))
        if job.attempts < 3:
            delay = _job_backoff(job.attempts)
            log(f"job failed ({job.kind}), retrying in {delay:.2f}s: {e}", "WARN")
            time.sleep(delay)
            _push_job(job)
        else:
            _m_inc(("jobs_failed_total", job.kind if job.kind in ("single","sweep") else "single"))
            log(f"job permanently failed after {job.attempts} attempts: {e}", "ERROR")

def _worker_loop(idx: int):
    log(f"worker {idx} started", "INFO")
    while True:
        job = JOB_Q.get()
        try:
            _run_job(job)
        finally:
            JOB_Q.task_done()

def _start_workers(n: int):
    if _WORKERS: 
        return
    for i in range(max(1, n)):
        t = threading.Thread(target=_worker_loop, args=(i+1,), daemon=True)
        _WORKERS.append(t)
        t.start()

# -------------------------
# Status Dashboard & Routes
# -------------------------
@app.get("/")
def status_dashboard():
    with _METRICS_LOCK:
        m = _METRICS.copy()
        jp = m["jobs_processed_total"].copy()
        jf = m["jobs_failed_total"].copy()
    
    uptime = int(max(0, time.time() - m["start_ts"]))
    qsize = JOB_Q.qsize() if JOB_Q else 0
    workers_ok = bool(_WORKERS) and (qsize < JOB_QUEUE_MAX)
    status_color = "#10b981" if workers_ok else "#ef4444"
    status_text = "Operational" if workers_ok else "Degraded"
    idemp_backend = "Redis" if IDEMP.redis is not None else "In-Memory"
    
    is_production = len(_WORKERS) > 0 and os.environ.get("REPL_ID")
    metrics_note = "Note: Multi-worker deployment - metrics show current worker only" if is_production else ""
    
    def format_uptime(seconds):
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NFT Sales Bot - Status</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 2rem;
                color: #fff;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            .header {{
                text-align: center;
                margin-bottom: 3rem;
            }}
            .logo {{
                width: 120px;
                height: 120px;
                border-radius: 50%;
                margin: 0 auto 1.5rem;
                display: block;
                box-shadow: 0 8px 24px rgba(0,0,0,0.3);
                border: 4px solid rgba(255,255,255,0.3);
            }}
            .header h1 {{
                font-size: 2.5rem;
                margin-bottom: 0.25rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                font-weight: 700;
            }}
            .header h2 {{
                font-size: 1.8rem;
                margin-bottom: 0.5rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                font-weight: 500;
                opacity: 0.95;
            }}
            .header p {{
                font-size: 1.1rem;
                opacity: 0.9;
            }}
            .status-badge {{
                display: inline-block;
                background: {status_color};
                padding: 0.5rem 1.5rem;
                border-radius: 2rem;
                font-weight: 600;
                margin: 1rem 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin-bottom: 2rem;
            }}
            .card {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            .card h3 {{
                font-size: 0.9rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                opacity: 0.8;
                margin-bottom: 0.5rem;
            }}
            .card .value {{
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }}
            .card .label {{
                font-size: 0.85rem;
                opacity: 0.7;
            }}
            .section {{
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 1rem;
                padding: 2rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }}
            .section h2 {{
                font-size: 1.5rem;
                margin-bottom: 1.5rem;
                border-bottom: 2px solid rgba(255, 255, 255, 0.2);
                padding-bottom: 0.5rem;
            }}
            .stat-row {{
                display: flex;
                justify-content: space-between;
                padding: 0.75rem 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .stat-row:last-child {{
                border-bottom: none;
            }}
            .stat-label {{
                opacity: 0.8;
            }}
            .stat-value {{
                font-weight: 600;
            }}
            .footer {{
                text-align: center;
                margin-top: 3rem;
                opacity: 0.7;
                font-size: 0.9rem;
            }}
            .endpoints {{
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
                margin-top: 1rem;
            }}
            .endpoint {{
                background: rgba(255, 255, 255, 0.15);
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                font-family: monospace;
                font-size: 0.85rem;
            }}
            .endpoint a {{
                color: #fff;
                text-decoration: none;
            }}
            .endpoint a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <img src="/static/images/gvc-logo.jpg" alt="Good Vibes Club Logo" class="logo">
                <h1>Good Vibes Club</h1>
                <h2>NFT Sales Bot</h2>
                <p>Automated NFT sales monitoring & Twitter posting</p>
                <div class="status-badge">● {status_text}</div>
            </div>

            <div class="grid">
                <div class="card">
                    <h3>Uptime</h3>
                    <div class="value">{format_uptime(uptime)}</div>
                    <div class="label">Running since startup</div>
                </div>
                <div class="card">
                    <h3>Workers</h3>
                    <div class="value">{len(_WORKERS)}</div>
                    <div class="label">Background threads active</div>
                </div>
                <div class="card">
                    <h3>Queue Depth</h3>
                    <div class="value">{qsize}/{JOB_QUEUE_MAX}</div>
                    <div class="label">Jobs in queue</div>
                </div>
                <div class="card">
                    <h3>Tweets Posted</h3>
                    <div class="value">{m['tweets_posted_total']}</div>
                    <div class="label">Total sent to Twitter/X</div>
                </div>
            </div>

            <div class="section">
                <h2>📊 Job Statistics</h2>
                <div class="stat-row">
                    <span class="stat-label">Single Sales Processed</span>
                    <span class="stat-value">{jp.get('single', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Sweep Sales Processed</span>
                    <span class="stat-value">{jp.get('sweep', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Single Sales Failed</span>
                    <span class="stat-value">{jf.get('single', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Sweep Sales Failed</span>
                    <span class="stat-value">{jf.get('sweep', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Media Uploads</span>
                    <span class="stat-value">{m['media_uploads_total']}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Media Upload Failures</span>
                    <span class="stat-value">{m['media_uploads_failed_total']}</span>
                </div>
            </div>

            <div class="section">
                <h2>⚙️ System Information</h2>
                <div class="stat-row">
                    <span class="stat-label">Idempotency Backend</span>
                    <span class="stat-value">{idemp_backend}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Collections Monitored</span>
                    <span class="stat-value">{len(COLLECTIONS) if COLLECTIONS else "All"}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Job Queue Max</span>
                    <span class="stat-value">{JOB_QUEUE_MAX}</span>
                </div>
            </div>

            <div class="section">
                <h2>🔗 API Endpoints</h2>
                <div class="endpoints">
                    <div class="endpoint"><a href="/health">/health</a></div>
                    <div class="endpoint"><a href="/ready">/ready</a></div>
                    <div class="endpoint"><a href="/metrics">/metrics</a></div>
                    <div class="endpoint"><a href="/metrics/prom">/metrics/prom</a></div>
                    <div class="endpoint">/webhook (POST)</div>
                </div>
            </div>

            <div class="footer">
                <p>NFT Sales Bot v2.0.0 • Powered by Moralis & Twitter/X API</p>
                <p style="margin-top: 0.5rem; opacity: 0.6;">Auto-refresh this page to see live updates</p>
                {f'<p style="margin-top: 0.5rem; opacity: 0.7; font-size: 0.85rem;">⚠️ {metrics_note}</p>' if metrics_note else ''}
            </div>
        </div>
    </body>
    </html>
    """
    return app.response_class(response=html, status=200, mimetype="text/html")

@app.get("/static/<path:filename>")
def serve_static(filename):
    from flask import send_from_directory
    return send_from_directory("static", filename)

@app.get("/health")
def health():
    return "ok", 200

@app.get("/ready")
def ready():
    ok = bool(_WORKERS) and (JOB_Q.qsize() < JOB_QUEUE_MAX)
    return ("ok", 200) if ok else ("busy", 503)

@app.get("/metrics")
def metrics_json():
    with _METRICS_LOCK:
        snap = json.loads(json.dumps(_METRICS))
    uptime = int(max(0, time.time() - snap["start_ts"]))
    qsize = JOB_Q.qsize() if JOB_Q else 0
    idemp_backend = "redis" if IDEMP.redis is not None else "memory"
    idemp_keys = None if IDEMP.redis is not None else len(IDEMP.mem)
    body = {
        "ok": True,
        "uptime_seconds": uptime,
        "queue_depth": qsize,
        "queue_max": JOB_QUEUE_MAX,
        "workers": len(_WORKERS),
        "idempotency_backend": idemp_backend,
        "idempotency_keys": idemp_keys,
        "metrics": snap,
    }
    return app.response_class(
        response=json.dumps(body, separators=(",",":")),
        status=200,
        mimetype="application/json"
    )

@app.get("/metrics/prom")
def metrics_prom():
    with _METRICS_LOCK:
        m = _METRICS.copy()
        jp = m["jobs_processed_total"].copy()
        jf = m["jobs_failed_total"].copy()
        jr = m["jobs_retried_total"].copy()
        mu = m["media_uploads_total"]
        muf = m["media_uploads_failed_total"]
        tp = m["tweets_posted_total"]
        start_ts = m["start_ts"]
    uptime = int(max(0, time.time() - start_ts))
    qsize = JOB_Q.qsize() if JOB_Q else 0
    idemp_backend = 1 if IDEMP.redis is not None else 0
    lines = [
        f'bot_uptime_seconds {uptime}',
        f'bot_queue_depth {qsize}',
        f'bot_workers {len(_WORKERS)}',
        f'bot_idempotency_backend{{backend="redis"}} {idemp_backend}',
        f'bot_media_uploads_total {mu}',
        f'bot_media_uploads_failed_total {muf}',
        f'bot_tweets_posted_total {tp}',
        f'bot_jobs_enqueued_total {m["jobs_enqueued_total"]}',
        f'bot_jobs_processed_total{{kind="single"}} {jp.get("single",0)}',
        f'bot_jobs_processed_total{{kind="sweep"}} {jp.get("sweep",0)}',
        f'bot_jobs_failed_total{{kind="single"}} {jf.get("single",0)}',
        f'bot_jobs_failed_total{{kind="sweep"}} {jf.get("sweep",0)}',
        f'bot_jobs_retried_total{{kind="single"}} {jr.get("single",0)}',
        f'bot_jobs_retried_total{{kind="sweep"}} {jr.get("sweep",0)}',
    ]
    return app.response_class(response="\n".join(lines) + "\n", status=200, mimetype="text/plain; version=0.0.4")

@app.post("/webhooks/moralis")
def moralis_webhook():
    if request.content_length and request.content_length > MAX_WEBHOOK_BYTES:
        return Response("Payload too large", status=413, mimetype="text/plain")

    raw = request.get_data(cache=False, as_text=False)
    sig = request.headers.get("x-signature") or request.headers.get("X-Signature") or ""
    if not verify_moralis_sig(raw, sig):
        return Response("Invalid signature", status=401, mimetype="text/plain")

    # Tolerant parsing so Moralis "Deploy" succeeds even if probe body is odd
    payload = request.get_json(silent=True)
    if payload is None and (request.headers.get("Content-Type","").lower().startswith("application/json")):
        try:
            payload = json.loads((raw or b"").decode("utf-8", errors="ignore"))
        except Exception:
            payload = None
    if payload is None:
        log(f"Webhook received non-JSON (CT={request.headers.get('Content-Type')})", "INFO")
        return "ok", 200

    if not payload.get("confirmed", True):
        return "ok", 200

    transfers = parse_nft_transfers(payload)
    if not transfers:
        return "ok", 200

    from collections import defaultdict
    buys = defaultdict(list)
    for t in transfers:
        if t["to"] and t["from"]:
            key = (t["tx"], t["to"], t["token_address"])
            buys[key].append(t)

    for (tx, buyer, token_address), items in buys.items():
        if IDEMP.seen(tx, buyer, token_address):
            log(f"skip duplicate {tx[:10]}… {shorten_addr(buyer)} {token_address[:8]}…", "INFO")
            continue
        
        tx_value = estimate_tx_total_eth(payload, tx)
        if tx_value is None or tx_value < 0.001:
            log(f"skip transfer (no ETH payment): {tx[:10]}… {shorten_addr(buyer)} {len(items)} NFT(s)", "INFO")
            continue
        
        IDEMP.mark(tx, buyer, token_address)

        items_sorted = sorted(items, key=lambda z: int(z["token_id"]))
        if len(items_sorted) >= 2:
            ok = _push_job(PostJob(kind="sweep", data={
                "buyer": buyer, "token_address": token_address,
                "items": items_sorted, "payload": payload
            }))
            if not ok: log("enqueue sweep failed (queue full)", "WARN")
        else:
            it = items_sorted[0]
            price_eth = None
            if it.get("value") is not None:
                try: price_eth = int(it["value"]) / 1e18
                except: price_eth = None
            if price_eth is None:
                price_eth = tx_value
            ok = _push_job(PostJob(kind="single", data={
                "contract": it["token_address"], "token_id": it["token_id"],
                "buyer": buyer, "seller": it["from"],
                "price_eth": price_eth, "payload": payload
            }))
            if not ok: log("enqueue single failed (queue full)", "WARN")

    return Response("accepted", status=202, mimetype="text/plain")

# -------------------------
# Debug routes
# -------------------------
@app.post("/debug/single")
def dbg_single():
    b = request.get_json(force=True)
    contract = b["contract"].lower()
    tid = str(b["token_id"])
    buyer = b["buyer"]; seller = b.get("seller", "0xseller...")
    price_eth = b.get("price_eth")
    fake_payload = {"nft_metadata": b.get("nft_metadata", [])}
    _push_job(PostJob(kind="single", data={
        "contract": contract, "token_id": tid, "buyer": buyer,
        "seller": seller, "price_eth": price_eth, "payload": fake_payload
    }))
    return "queued", 202

@app.post("/debug/sweep")
def dbg_sweep():
    b = request.get_json(force=True)
    buyer = b["buyer"].lower()
    items_in = b["items"]  # [{contract, token_id, price_eth?, seller?}, ...] (same contract for sweep)
    contract = items_in[0]["contract"].lower()
    norm = []
    for it in items_in:
        norm.append({
            "tx": "0xdebug",
            "to": buyer,
            "from": it.get("seller","0xseller...").lower(),
            "token_address": contract,
            "token_id": str(it["token_id"]),
            "value": int(float(it.get("price_eth", 0))*1e18) if it.get("price_eth") is not None else None
        })
    fake_payload = {"nft_metadata": b.get("nft_metadata", [])}
    _push_job(PostJob(kind="sweep", data={
        "buyer": buyer, "token_address": contract, "items": norm, "payload": fake_payload
    }))
    return "queued", 202

# -------------------------
# Initialize workers (for both dev and production)
# -------------------------
watching = ", ".join(sorted(COLLECTIONS)) if COLLECTIONS else "(all)"
log(f"Watching collections: {watching}")
_start_workers(JOB_WORKERS)

# -------------------------
# Main (development server only)
# -------------------------
if __name__ == "__main__":
    log("Starting Flask development server on 0.0.0.0:{PORT}".replace("{PORT}", str(PORT)))
    app.run(host="0.0.0.0", port=PORT, debug=False)
