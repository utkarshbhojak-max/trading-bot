import os
import json
import asyncio
import logging
import tempfile
import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# --- LOGGING ---
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("trading_bot")

# --- CONFIG ---
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise SystemExit(
        "TELEGRAM_TOKEN is not set. Create a .env file with:\n"
        "TELEGRAM_TOKEN=your_bot_token_here\n"
        "Never hardcode real tokens in source code."
    )

SUBSCRIBERS_FILE = "subscribers.json"
_subscriber_lock = asyncio.Lock()

PRO_RISK_PER_TRADE_INR = 2000
MIN_DAILY_TURNOVER_INR = 30_000_000
TARGET_1_PCT = 0.05
TARGET_2_PCT = 0.09
MIN_BARS_REQUIRED = 210  # need > 200 for a meaningfully warmed-up EMA200
MSG_SEND_DELAY_SEC = 0.6  # avoid Telegram flood limits when sending multiple signals

WATCHLIST = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS",
    "LT.NS", "ITC.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "NESTLEIND.NS", "ASIANPAINT.NS", "TATASTEEL.NS",
    "POWERGRID.NS", "TECHM.NS", "BAJAJFINSV.NS", "GRASIM.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "ADANIENT.NS", "DIVISLAB.NS", "M&M.NS", "APOLLOHOSP.NS",
    "BPCL.NS", "COALINDIA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS"
]

# --- SUBSCRIBER MANAGEMENT (atomic + lock-safe) ---
def _load_subscribers_sync() -> set:
    if os.path.exists(SUBSCRIBERS_FILE):
        try:
            with open(SUBSCRIBERS_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            logger.exception("Failed to read %s, starting fresh", SUBSCRIBERS_FILE)
            return set()
    return set()

def _write_subscribers_sync(subs: set) -> None:
    # Atomic write: write to temp file then rename, so a crash mid-write
    # never corrupts subscribers.json
    dir_name = os.path.dirname(os.path.abspath(SUBSCRIBERS_FILE)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(list(subs), f)
        os.replace(tmp_path, SUBSCRIBERS_FILE)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

async def load_subscribers() -> set:
    return await asyncio.to_thread(_load_subscribers_sync)

async def add_subscriber(chat_id: int) -> bool:
    """Returns True if newly added, False if already subscribed."""
    async with _subscriber_lock:
        subs = await asyncio.to_thread(_load_subscribers_sync)
        if chat_id in subs:
            return False
        subs.add(chat_id)
        await asyncio.to_thread(_write_subscribers_sync, subs)
        return True

async def remove_subscriber(chat_id: int) -> bool:
    """Returns True if removed, False if wasn't subscribed."""
    async with _subscriber_lock:
        subs = await asyncio.to_thread(_load_subscribers_sync)
        if chat_id not in subs:
            return False
        subs.discard(chat_id)
        await asyncio.to_thread(_write_subscribers_sync, subs)
        return True

# --- TECHNICAL INDICATORS ---
def get_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's smoothing RSI (the industry-standard version used by
    TradingView / most brokers) rather than a simple rolling-mean RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)  # avg_loss == 0 means pure uptrend -> RSI 100

def get_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    # Wilder's smoothing for ATR, consistent with the RSI change above
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

def get_macd(series: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # MACD histogram

def fetch_data_blocking():
    return yf.download(
        WATCHLIST, period="14mo", interval="1d", group_by="ticker",
        progress=False, auto_adjust=True,
    )

# --- SCANNER CORE ---
async def run_market_scan() -> list[dict]:
    try:
        df = await asyncio.to_thread(fetch_data_blocking)
    except Exception:
        logger.exception("yfinance download failed")
        return []

    if df is None or df.empty:
        logger.warning("Market data download returned empty dataframe")
        return []

    is_multi = isinstance(df.columns, pd.MultiIndex)
    available_tickers = set(df.columns.get_level_values(0).unique()) if is_multi else set(WATCHLIST[:1])

    candidates = []
    for ticker in WATCHLIST:
        try:
            if is_multi:
                if ticker not in available_tickers:
                    continue
                hist = df[ticker].dropna(subset=["Close", "High", "Low", "Volume"])
            else:
                hist = df.dropna(subset=["Close", "High", "Low", "Volume"])

            if hist.empty or len(hist) < MIN_BARS_REQUIRED:
                continue

            close = hist["Close"]
            price = float(close.iloc[-1])

            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

            rsi = float(get_rsi(close).iloc[-1])
            atr = float(get_atr(hist).iloc[-1])
            macd_hist = float(get_macd(close).iloc[-1])
            curr_vol = float(hist["Volume"].iloc[-1])
            avg_vol = float(hist["Volume"].rolling(window=20).mean().iloc[-1])

            if any(pd.isna(v) for v in (rsi, atr, macd_hist)) or avg_vol == 0:
                continue

            if (price * avg_vol) < MIN_DAILY_TURNOVER_INR:
                continue

            vol_ratio = curr_vol / avg_vol
            score = 0
            direction = "BULLISH" if rsi >= 50 else "BEARISH"

            if direction == "BULLISH":
                if price > ema200: score += 25
                if ema20 > ema50: score += 20
                if rsi >= 52: score += 20
                if macd_hist > 0: score += 15
                if vol_ratio >= 1.0: score += 20
            else:
                if price < ema200: score += 25
                if ema20 < ema50: score += 20
                if rsi <= 48: score += 20
                if macd_hist < 0: score += 15
                if vol_ratio >= 1.0: score += 20

            if score < 60:
                continue

            tier_tag = "🟢 TIER 1: INST. SWING" if score >= 80 else "⚡ TIER 2: SETUP WATCHLIST"
            max_stop_pct = price * 0.035
            sl_dist = min(1.5 * atr, max_stop_pct)
            if sl_dist <= 0:
                continue

            if direction == "BULLISH":
                entry = price
                sl = entry - sl_dist
                t1 = entry * (1 + TARGET_1_PCT)
                t2 = entry * (1 + TARGET_2_PCT)
                risk_per_share = entry - sl
            else:
                entry = price
                sl = entry + sl_dist
                t1 = entry * (1 - TARGET_1_PCT)
                t2 = entry * (1 - TARGET_2_PCT)
                risk_per_share = sl - entry

            if risk_per_share <= 0:
                continue

            pro_qty = int(PRO_RISK_PER_TRADE_INR / risk_per_share)
            if pro_qty < 1:
                continue  # position sizing would round to 0 shares; not tradeable at this risk budget

            candidates.append({
                "ticker": ticker.replace(".NS", ""),
                "direction": direction,
                "signal": tier_tag,
                "score": score,
                "rsi": round(rsi, 2),
                "vol_ratio": round(vol_ratio, 2),
                "entry": entry,
                "sl": sl,
                "t1": t1,
                "t2": t2,
                "pro_qty": pro_qty,
                "pro_cap": pro_qty * entry,
                "profit_1_share": abs(t1 - entry),
                "qty_5k": max(1, int(5000 // entry)),
                "qty_10k": max(1, int(10000 // entry)),
            })
        except Exception:
            logger.exception("Error processing ticker %s", ticker)
            continue

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:5]

async def send_scan_report(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    candidates = await run_market_scan()
    if not candidates:
        await context.bot.send_message(
            chat_id=chat_id,
            text="⚠️ *Market Update:* No high-probability 5%+ swing setups passed today's filter.",
            parse_mode="Markdown",
        )
        return

    for r in candidates:
        arrow = "🟩" if r["direction"] == "BULLISH" else "🟥"
        msg = (
            f"{arrow} *{r['ticker']}*  `{r['score']}/100 Score`  ({r['direction']})\n"
            f"_{r['signal']}_\n"
            f"───\n"
            f"📍 *Entry:* ₹{r['entry']:,.2f}\n"
            f"🎯 *Target 1 (±5%):* ₹{r['t1']:,.2f}\n"
            f"🚀 *Target 2 (±9%):* ₹{r['t2']:,.2f}\n"
            f"🛑 *Stop Loss:* ₹{r['sl']:,.2f}\n"
            f"⏳ *Hold Time:* 3–8 Days\n\n"
            f"🎓 *Student Sizing (Small Budget)*\n"
            f"└ `1 Share` ➜ ₹{r['entry']:,.2f} *(±₹{r['profit_1_share']:,.2f} at T1)*\n"
            f"└ `₹5k Plan` ➜ {r['qty_5k']} share(s)\n"
            f"└ `₹10k Plan` ➜ {r['qty_10k']} share(s)\n\n"
            f"💼 *Pro Sizing (Risk ₹{PRO_RISK_PER_TRADE_INR})*\n"
            f"└ {r['pro_qty']} shares • Capital: ₹{r['pro_cap']:,.2f}\n\n"
            f"_Not financial advice — a mechanical setup, not a guarantee. Size positions to what you can afford to lose._"
        )
        try:
            await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
        except Exception:
            logger.exception("Failed to send message to chat_id=%s", chat_id)
        await asyncio.sleep(MSG_SEND_DELAY_SEC)

# --- COMMAND HANDLERS ---
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    newly_added = await add_subscriber(chat_id)

    welcome_text = (
        "🚀 *Welcome to Pro Market Scanner!*\n\n"
        + ("✅ You are now subscribed to **Daily Automatic Alerts (3:15 PM IST, Mon–Fri)**.\n"
           if newly_added else "ℹ️ You're already subscribed to Daily Alerts.\n")
        + "⚡ *Scanning market now — this can take up to a minute...*"
    )
    await update.message.reply_text(welcome_text, parse_mode="Markdown")
    await send_scan_report(context, chat_id)

async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manual on-demand scan, independent of subscription status."""
    chat_id = update.effective_chat.id
    await update.message.reply_text("⚡ Scanning market now — this can take up to a minute...")
    await send_scan_report(context, chat_id)

async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    removed = await remove_subscriber(chat_id)
    text = "✅ Unsubscribed from daily alerts." if removed else "You weren't subscribed."
    await update.message.reply_text(text)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Unhandled exception while processing update: %s", update, exc_info=context.error)

async def scheduled_daily_broadcast(context: ContextTypes.DEFAULT_TYPE):
    subscribers = await load_subscribers()
    logger.info("Running scheduled broadcast for %d subscribers", len(subscribers))
    for chat_id in subscribers:
        try:
            await send_scan_report(context, chat_id)
        except Exception:
            logger.exception("Broadcast failed for chat_id=%s", chat_id)
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("scan", scan_cmd))
    app.add_handler(CommandHandler("stop", stop_cmd))
    app.add_error_handler(error_handler)

    target_time = datetime.time(hour=15, minute=15, second=0, tzinfo=ZoneInfo("Asia/Kolkata"))

    if app.job_queue:
        app.job_queue.run_daily(
            scheduled_daily_broadcast,
            time=target_time,
            days=(1, 2, 3, 4, 5),
        )
    else:
        logger.warning(
            "JobQueue not available — install with: pip install \"python-telegram-bot[job-queue]\""
        )

    logger.info("Trading Bot live. Commands: /start (subscribe+scan), /scan (on-demand), /stop (unsubscribe)")
    app.run_polling()