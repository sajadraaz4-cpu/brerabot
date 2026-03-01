import math
import json
import csv
import os
import re
import time
import logging
from datetime import datetime, date

import requests
from flask import Flask, render_template, Response, session, request, redirect, url_for, g

# ---------------------------------------------------------------------------
# Gemini (google-genai) — importazione sicura
# ---------------------------------------------------------------------------
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_ACTIVES_URL = "https://financialmodelingprep.com/stable/most-actives"
FMP_GAINERS_URL = "https://financialmodelingprep.com/stable/biggest-gainers"
API_DAILY_LIMIT = 250
USAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_usage.json")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risultati")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("investbot")

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "brera_bot_s3cr3t")
APP_PIN = os.environ.get("APP_PIN", "2026")

# ---------------------------------------------------------------------------
# API Usage Tracker
# ---------------------------------------------------------------------------
def load_api_usage() -> dict:
    today = date.today().isoformat()
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    return {"date": today, "count": 0}

def save_api_usage(data: dict) -> None:
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

# ---------------------------------------------------------------------------
# AI Score (Value Investing × Metodo Brera — 0‑100)
# ---------------------------------------------------------------------------
_JUNK_KEYWORDS = {
    "etf", "fund", "trust", "direxion", "proshares",
    "long", "short", "inverse", "leveraged", "ultra",
    "3x", "2x", "-1x", "bear", "bull",
}

_TROPHY_ASSETS_WHITELIST = {
    "GLD", "IAU", "SLV", "BTC", "IBIT", "GBTC",
}

def compute_ai_score(stock: dict) -> float:
    symbol = str(stock.get("symbol") or "").strip().upper()
    score = 0.0

    if symbol in _TROPHY_ASSETS_WHITELIST:
        score += 60.0

    pe_raw = stock.get("peRatio")
    try:
        pe = float(pe_raw)
    except (ValueError, TypeError):
        pe = None

    if pe is not None and 0 < pe <= 25:
        if pe >= 10:
            distance = abs(pe - 17.0)
            score += max(15.0, 25.0 - distance * 1.0)
        else:
            score += max(5.0, 15.0 - (10.0 - pe) * 1.5)

    eps_raw = stock.get("eps")
    try:
        eps = float(eps_raw)
    except (ValueError, TypeError):
        eps = None

    if eps is not None and eps > 0:
        eps_pts = math.log1p(eps) * 5.0
        score += min(15.0, eps_pts)

    change_pct_raw = stock.get("changesPercentage") or 0
    try:
        change_pct = float(change_pct_raw)
    except (ValueError, TypeError):
        change_pct = 0.0

    if change_pct < -3.0:
        score += max(-20.0, change_pct * 2.0)
    elif -3.0 <= change_pct < 0.0:
        score += change_pct * 1.0
    elif 0.0 <= change_pct <= 2.0:
        score += change_pct * 3.0
    elif 2.0 < change_pct <= 8.0:
        score += 6.0 + (change_pct - 2.0) * 2.3333
    elif 8.0 < change_pct <= 15.0:
        score += max(5.0, 20.0 - (change_pct - 8.0) * 2.14)
    else:
        score -= min(20.0, (change_pct - 15.0) * 3.0)

    beta_raw = stock.get("beta")
    try:
        beta = float(beta_raw)
    except (ValueError, TypeError):
        beta = None

    if beta is not None:
        if 0.5 <= beta <= 1.2:
            score += 10.0
        elif beta < 0.5:
            score += 5.0
        elif 1.2 < beta <= 1.5:
            score += 3.0
        elif 1.5 < beta <= 1.8:
            score -= 3.0
        else:
            score -= 10.0

    div_yield = stock.get("dividendYield")
    if div_yield is None:
        last_div = stock.get("lastDiv")
        price_for_div = stock.get("price") or 0
        try:
            price_for_div = float(price_for_div)
            last_div = float(last_div) if last_div is not None else 0.0
            div_yield = (last_div / price_for_div * 100) if price_for_div > 0 else 0.0
        except (ValueError, TypeError, ZeroDivisionError):
            div_yield = 0.0
    else:
        try:
            div_yield = float(div_yield)
        except (ValueError, TypeError):
            div_yield = 0.0

    if div_yield > 0:
        if div_yield <= 5.0:
            score += div_yield * 2.0
        else:
            score += max(6.0, 10.0 - (div_yield - 5.0) * 0.5)

    return round(max(0.0, min(100.0, score)), 1)


# ---------------------------------------------------------------------------
# CSV Export & Gemini
# ---------------------------------------------------------------------------
def save_results_csv(stocks: list[dict], scores: list[float]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"risultati_{now}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)
    fieldnames = ["rank", "symbol", "name", "exchange", "price", "change", "changesPercentage", "peRatio", "eps", "beta", "dividendYield", "ai_score"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (stock, score) in enumerate(zip(stocks, scores), start=1):
            writer.writerow({
                "rank": i,
                "symbol": stock.get("symbol", "N/A"),
                "name": stock.get("name", "N/A"),
                "exchange": stock.get("exchange", "N/A"),
                "price": stock.get("price", 0),
                "change": stock.get("change", 0),
                "changesPercentage": stock.get("changesPercentage", 0),
                "peRatio": stock.get("peRatio", "N/A"),
                "eps": stock.get("eps", "N/A"),
                "beta": stock.get("beta", "N/A"),
                "dividendYield": round(stock.get("dividendYield") or 0, 2),
                "ai_score": score,
            })
    return filename


def generate_gemini_comment(stock: dict, score: float):
    if not GEMINI_AVAILABLE: return None
    if not GEMINI_API_KEY: return "⚠ Errore: Chiave API Gemini mancante."
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        symbol = stock.get("symbol", "N/A")
        name = stock.get("name", "N/A")
        price = stock.get("price", 0)
        pe = stock.get("peRatio", "N/A")
        eps = stock.get("eps", "N/A")
        change_pct = stock.get("changesPercentage", 0)
        beta = stock.get("beta", "N/A")
        div_yield = stock.get("dividendYield", 0)
        is_trophy = symbol.upper() in _TROPHY_ASSETS_WHITELIST
        asset_type = "Trophy Asset Anti-Debasement" if is_trophy else "Equity Value"

        prompt = (f"Sei Guido Maria Brera. Analizza {name} ({symbol}). "
                  f"Tipologia: {asset_type}. Dati: Prezzo ${price}, P/E {pe}, EPS {eps}, "
                  f"Beta {beta}, Div. Yield {div_yield}%, Variazione {change_pct}%, AI Score {score}/100.\n"
                  f"Dai un giudizio narrativo, colto e tagliente in esattamente 4 frasi, "
                  f"basandoti sul debasement monetario, i trophy asset e l'approccio contrarian. "
                  f"Tono sofisticato, in italiano.")

        for attempt in range(3):
            try:
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                return response.text.strip() if response and response.text else None
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < 2: time.sleep(2 * (attempt + 1)); continue
                    return "⚠ [Quota AI Esaurita] — Riprova più tardi."
                return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Rotte
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in") and not request.args.get("force"):
        return redirect(url_for("index"))
    error = None
    if request.method == "POST":
        wants_json = request.is_json or "json" in request.headers.get("Accept", "")
        pin = ""
        try:
            pin = str(request.get_json().get("pin", "")).strip() if request.is_json else str(request.form.get("pin", "")).strip()
        except: pass
        if pin == APP_PIN.strip():
            session["logged_in"] = True
            return {"success": True} if wants_json else redirect(url_for("index"))
        else:
            error = "PIN errato."
            if wants_json: return {"success": False, "error": error}, 401
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def index():
    if not session.get("logged_in"): return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/about")
def about():
    if not session.get("logged_in"): return redirect(url_for("login"))
    return render_template("about.html")

@app.route("/api/analyze")
def analyze():
    if not session.get("logged_in"):
        return Response((sse_event({"type": "error", "message": "⛔ Accesso negato."}) for _ in range(1)), mimetype="text/event-stream")
    import requests

    def generate():
        yield sse_event({"type": "info", "message": "➜ ~ $ python avvia_agente.py --mode=live"})
        time.sleep(0.4)
        yield sse_event({"type": "info", "message": "[+] Inizializzazione Core System v3.0.0..."})
        
        usage = load_api_usage()
        if (API_DAILY_LIMIT - usage["count"]) < 25:
            yield sse_event({"type": "error", "message": "✖ ERRORE: Limite API giornaliero FMP insufficiente."})
            return

        yield sse_event({"type": "info", "message": "> Connessione al modello logico..."})
        time.sleep(0.3)
        yield sse_event({"type": "info", "message": "> Scaricamento dati di mercato..."})

        stocks = []
        try:
            resp1 = requests.get(FMP_ACTIVES_URL, params={"apikey": FMP_API_KEY}, timeout=15)
            if resp1.status_code == 200: stocks.extend(resp1.json())
            usage["count"] += 1
            
            resp2 = requests.get(FMP_GAINERS_URL, params={"apikey": FMP_API_KEY}, timeout=15)
            if resp2.status_code == 200: stocks.extend(resp2.json())
            usage["count"] += 1

            # Deduplica
            seen, unique_stocks = set(), []
            for s in stocks:
                if isinstance(s, dict) and s.get("symbol") and s["symbol"] not in seen:
                    seen.add(s["symbol"])
                    unique_stocks.append(s)
            stocks = unique_stocks

            yield sse_event({"type": "info", "message": "  [1/3] Fetching Fundamental Data (Quote & Profile)..."})
            
            # Filtro base
            def _is_junk(s):
                sym = str(s.get("symbol", "")).upper()
                if sym in _TROPHY_ASSETS_WHITELIST: return False
                nm = str(s.get("name", "")).lower()
                return any(kw in nm or kw in sym.lower() for kw in _JUNK_KEYWORDS)

            stocks = [s for s in stocks if not _is_junk(s)]
            stocks.sort(key=lambda s: abs(s.get("changesPercentage", 0) or 0) * (s.get("price", 0) or 0), reverse=True)
            top_symbols = [s.get("symbol") for s in stocks[:10] if s.get("symbol")]

            profile_map = {}
            profile_api_calls = 0

            for sym in top_symbols:
                if (usage["count"] + profile_api_calls) >= API_DAILY_LIMIT - 2: break
                try:
                    combined = {}
                    # 1. QUOTE per PE e EPS
                    r_q = requests.get(f"https://financialmodelingprep.com/api/v3/quote/{sym}", params={"apikey": FMP_API_KEY}, timeout=10)
                    profile_api_calls += 1
                    if r_q.status_code == 200 and r_q.json():
                        combined["peRatio"] = r_q.json()[0].get("pe")
                        combined["eps"] = r_q.json()[0].get("eps")

                    # 2. PROFILE per Beta e Dividendi
                    r_p = requests.get(f"https://financialmodelingprep.com/api/v3/profile/{sym}", params={"apikey": FMP_API_KEY}, timeout=10)
                    profile_api_calls += 1
                    if r_p.status_code == 200 and r_p.json():
                        combined["beta"] = r_p.json()[0].get("beta")
                        combined["lastDiv"] = r_p.json()[0].get("lastDiv")
                        combined["dividendYield"] = r_p.json()[0].get("dividendYield")

                    profile_map[sym] = combined
                except Exception as e:
                    logger.error("Errore fetch dati per %s: %s", sym, e)

            usage["count"] += profile_api_calls
            save_api_usage(usage)

            final_stocks = []
            for s in stocks:
                sym = s.get("symbol", "")
                if sym in profile_map:
                    s.update(profile_map[sym])
                    final_stocks.append(s)
            stocks = final_stocks

        except Exception as e:
            yield sse_event({"type": "error", "message": f"✖ ERRORE di rete."})
            return

        yield sse_event({"type": "success", "message": f"  ↳ {len(stocks)} aziende verificate e pronte per lo scoring."})
        yield sse_event({"type": "info", "message": "> Analisi fondamentale in corso..."})

        scored = [(s, compute_ai_score(s)) for s in stocks]
        valid_scored = sorted([(s, sc) for s, sc in scored if sc > 0.0], key=lambda x: x[1], reverse=True)
        top10 = valid_scored[:10]
        scored.sort(key=lambda x: x[1], reverse=True)

        yield sse_event({"type": "success", "message": f"  ↳ {len(valid_scored)} superano i filtri ({len(scored) - len(valid_scored)} scartate)."})
        time.sleep(0.5)

        if not top10:
            yield sse_event({"type": "warning", "message": "⚠ Nessuna azienda ha superato i filtri."})
            save_results_csv([s for s, _ in scored], [sc for _, sc in scored])
            yield sse_event({"type": "complete", "message": "✔ Analisi completata."})
            return

        yield sse_event({"type": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"})
        yield sse_event({"type": "success", "message": "  🏆  TOP 10 — AI Investment Ranking"})
        yield sse_event({"type": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"})

        for rank, (stock, score) in enumerate(top10, start=1):
            symbol, name, price, change_pct = stock.get("symbol", "?"), stock.get("name", "N/A"), stock.get("price", 0), stock.get("changesPercentage", 0)
            pe, eps_val = stock.get("peRatio", "N/A"), stock.get("eps", "N/A")
            arrow = "↑" if change_pct >= 0 else "↓"
            yield sse_event({
                "type": "result", "message": f"  #{rank}  {symbol}",
                "detail": f"{name} | ${price:,.2f} | {arrow}{abs(change_pct):.2f}% | P/E: {pe} | EPS: {eps_val} | Score: {score}",
                "sector": stock.get("exchange", "N/A"),
            })
            time.sleep(0.3)

        yield sse_event({"type": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"})
        if top10: yield sse_event({"type": "chart_data", "payload": [{"symbol": s.get("symbol", "?"), "ai_score": sc} for s, sc in top10]})

        if top10:
            yield sse_event({"type": "info", "message": "> Collegamento a Gemini AI..."})
            gemini_text = generate_gemini_comment(top10[0][0], top10[0][1])
            if gemini_text:
                yield sse_event({"type": "info", "message": f"  📝 Commento AI su {top10[0][0].get('symbol', '?')}:"})
                for i in range(0, len(gemini_text), 8):
                    yield sse_event({"type": "gemini", "message": gemini_text[i:i+8]})
                    time.sleep(0.04)
                yield sse_event({"type": "gemini_end", "message": ""})
            else:
                yield sse_event({"type": "warning", "message": "  ⚠ AI non disponibile."})

        save_results_csv([s for s, _ in scored], [sc for _, sc in scored])
        yield sse_event({"type": "complete", "message": f"✔ Analisi completata con successo."})

    return Response(generate(), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
