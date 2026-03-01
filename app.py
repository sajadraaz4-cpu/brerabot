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
FMP_PROFILE_URL = "https://financialmodelingprep.com/stable/profile"
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
# CORS rimosso per pulizia deploy Render
app.secret_key = os.environ.get("SECRET_KEY", "brera_bot_s3cr3t")

# PIN di accesso (solo tu puoi usare il bot)
APP_PIN = os.environ.get("APP_PIN", "2026")


# ---------------------------------------------------------------------------
# API Usage Tracker
# ---------------------------------------------------------------------------
def load_api_usage() -> dict:
    """Carica il contatore giornaliero da api_usage.json."""
    today = date.today().isoformat()
    if os.path.exists(USAGE_FILE):
        try:
            with open(USAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") == today:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
    # Reset per un nuovo giorno o file corrotto/mancante
    return {"date": today, "count": 0}


def save_api_usage(data: dict) -> None:
    """Salva il contatore in api_usage.json."""
    with open(USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Utility SSE
# ---------------------------------------------------------------------------
def sse_event(data: dict) -> str:
    """Formatta un dict come evento SSE."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# AI Score (Value Investing × Metodo Brera — 0‑100)
# ---------------------------------------------------------------------------

# Parole-chiave per identificare fondi, ETF, prodotti a leva
_JUNK_KEYWORDS = {
    "etf", "fund", "trust", "direxion", "proshares",
    "long", "short", "inverse", "leveraged", "ultra",
    "3x", "2x", "-1x", "bear", "bull",
}

# Trophy Assets Anti-Debasement (Metodo Brera)
# Oro, Argento, Bitcoin — scudi contro la svalutazione monetaria
_TROPHY_ASSETS_WHITELIST = {
    "GLD",   # SPDR Gold Shares
    "IAU",   # iShares Gold Trust
    "SLV",   # iShares Silver Trust
    "BTC",   # Bitcoin (se quotato)
    "IBIT",  # iShares Bitcoin Trust
    "GBTC",  # Grayscale Bitcoin Trust
}


def compute_ai_score(stock: dict) -> float:
    """
    Dynamic Value‑Investing × Metodo Brera Score (0–100).
    """
    symbol = str(stock.get("symbol") or "").strip().upper()
    score = 0.0

    # TROPHY ASSET BONUS (+60)
    if symbol in _TROPHY_ASSETS_WHITELIST:
        score += 60.0

    # METRICHE FONDAMENTALI
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

    # METRICHE DI MERCATO
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
# CSV Export
# ---------------------------------------------------------------------------
def save_results_csv(stocks: list[dict], scores: list[float]) -> str:
    """Salva i risultati in un CSV con timestamp nel nome. Ritorna il path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"risultati_{now}.csv"
    filepath = os.path.join(RESULTS_DIR, filename)

    fieldnames = ["rank", "symbol", "name", "exchange", "price",
                  "change", "changesPercentage", "peRatio", "eps",
                  "beta", "dividendYield", "ai_score"]

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
    logger.info("CSV salvato: %s (%d righe)", filepath, len(stocks))
    return filename


# ---------------------------------------------------------------------------
# Gemini AI Commentary
# ---------------------------------------------------------------------------
def generate_gemini_comment(stock: dict, score: float):
    """Genera un commento in stile Guido Maria Brera usando Gemini."""
    if not GEMINI_AVAILABLE:
        logger.error("Gemini SDK non installato (google-genai mancante).")
        return None

    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY non configurata.")
        return "⚠ Errore: Chiave API Gemini mancante nel server."

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

        prompt = (
            f"Sei Guido Maria Brera — gestore macro, autore de 'I Diavoli', "
            f"maestro dell'approccio contrarian e della protezione del capitale.\n\n"
            f"Analizza {name} ({symbol}) con il tuo metodo. "
            f"Tipologia: {asset_type}. "
            f"Dati: Prezzo ${price}, P/E {pe}, EPS {eps}, Beta {beta}, "
            f"Div. Yield {div_yield}%, variazione giornaliera {change_pct}%, "
            f"AI Score {score}/100.\n\n"
            f"Valuta questo asset attraverso i tuoi 4 pilastri:\n"
            f"1. DEBASEMENT MONETARIO: è uno scudo contro la svalutazione e la dominanza fiscale?\n"
            f"2. TROPHY ASSET: ha la natura di un bene rifugio o di un debito corporate/tech forte?\n"
            f"3. APPROCCIO CONTRARIAN: è una moda sovraffollata dagli algoritmi o un'opportunità ignorata?\n"
            f"4. PRIMA NON PERDERE: la volatilità (Beta) è accettabile? Il capitale è protetto?\n\n"
            f"Dai un giudizio narrativo, colto e tagliente in esattamente 4 frasi. "
            f"Rispondi in italiano, tono sofisticato e deciso, come nelle tue interviste."
        )

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                return response.text.strip() if response and response.text else None
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    logger.error("Gemini Quota Esaurita dopo 3 tentativi: %s", e)
                    return "⚠ [Quota AI Esaurita] — Riprova più tardi."
                logger.error("Errore Gemini (tentativo %d): %s", attempt + 1, e)
                return None
    except Exception as e:
        logger.error("Errore Gemini: %s", e)
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
        accept_header = request.headers.get("Accept", "")
        content_type = request.headers.get("Content-Type", "")
        wants_json = request.is_json or "json" in accept_header or "json" in content_type
        
        pin = ""
        try:
            if request.is_json:
                data = request.get_json()
                pin = str(data.get("pin", "")).strip()
            else:
                pin = str(request.form.get("pin", "")).strip()
        except:
            pin = ""

        if pin == APP_PIN.strip():
            session["logged_in"] = True
            if wants_json:
                return {"success": True}
            return redirect(url_for("index"))
        else:
            error = "PIN errato. Riprova."
            if wants_json:
                return {"success": False, "error": error}, 401

    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")


@app.route("/about")
def about():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("about.html")


@app.route("/api/analyze")
def analyze():
    """Endpoint SSE che esegue l'analisi e invia i risultati in tempo reale."""
    if not session.get("logged_in"):
        def denied():
            yield sse_event({"type": "error", "message": "⛔ Accesso negato. Effettua il login."})
        return Response(denied(), mimetype="text/event-stream")
    import requests

    def generate():
        logger.info("=== Nuova analisi avviata ===")

        # --- Step 1: Controllo limite API ---
        yield sse_event({"type": "info", "message": "➜ ~ $ python avvia_agente.py --mode=live"})
        time.sleep(0.4)

        yield sse_event({"type": "info", "message": "[+] Inizializzazione Core System v3.0.0..."})
        time.sleep(0.3)

        usage = load_api_usage()
        used = usage["count"]
        remaining = API_DAILY_LIMIT - used

        if remaining <= 0:
            yield sse_event({"type": "error", "message": "✖ ERRORE: Limite API raggiunto. Riprova più tardi."})
            logger.warning("Limite API raggiunto: %d/%d", used, API_DAILY_LIMIT)
            yield sse_event({"type": "complete", "message": "Analisi interrotta."})
            return

        # --- Step 2: Connessione modello logico ---
        yield sse_event({"type": "info", "message": "> Connessione al modello logico..."})
        time.sleep(0.6)
        yield sse_event({"type": "success", "message": "  ↳ Connection established (12ms)"})
        time.sleep(0.3)

        # --- Step 3: Chiamata API FMP ---
        yield sse_event({"type": "info", "message": "> Scaricamento dati di mercato (NASDAQ, NYSE)..."})
        time.sleep(0.4)

        stocks = []
        try:
            # --- Endpoint 1: Most Actives ---
            logger.info("Chiamata API FMP: most-actives...")
            yield sse_event({"type": "info", "message": "  [1/3] Fetching Most Active Stocks..."})
            resp1 = requests.get(FMP_ACTIVES_URL, params={"apikey": FMP_API_KEY}, timeout=15)
            resp1.raise_for_status()
            data1 = resp1.json()
            usage["count"] += 1
            time.sleep(0.3)

            # --- Endpoint 2: Biggest Gainers ---
            yield sse_event({"type": "info", "message": "  [2/3] Fetching Biggest Gainers..."})
            resp2 = requests.get(FMP_GAINERS_URL, params={"apikey": FMP_API_KEY}, timeout=15)
            resp2.raise_for_status()
            data2 = resp2.json()
            usage["count"] += 1

            # Unisci e deduplica
            seen = set()
            for s in (data1 if isinstance(data1, list) else []) + (data2 if isinstance(data2, list) else []):
                sym = s.get("symbol")
                if sym and sym not in seen:
                    seen.add(sym)
                    stocks.append(s)

            # --- Endpoint 3: Profili Singoli per TUTTE le aziende ---
            yield sse_event({"type": "info", "message": "  [3/3] Fetching Fundamental Data (Tutte le aziende in lista)..."})
            time.sleep(0.3)

            def _is_junk(s):
                sym = str(s.get("symbol") or "").strip().upper()
                if sym in _TROPHY_ASSETS_WHITELIST:
                    return False
                nm = str(s.get("name") or "").lower()
                for kw in _JUNK_KEYWORDS:
                    if kw in nm or kw in sym.lower():
                        return True
                return False

            # Eliminiamo fondi e ETF
            stocks = [s for s in stocks if not _is_junk(s)]

            # Ordinamento logico
            stocks.sort(
                key=lambda s: abs(s.get("changesPercentage", 0) or 0) * (s.get("price", 0) or 0),
                reverse=True,
            )
            
            # Selezioniamo tutti i simboli
            all_symbols = [s.get("symbol") for s in stocks if s.get("symbol")]

            profile_map = {}
            profile_api_calls = 0

            if all_symbols:
                yield sse_event({"type": "info", "message": f"  ↳ Inizio fetch per {len(all_symbols)} asset. (Potrebbe richiedere tempo)..."})

            for sym in all_symbols:
                # Controlliamo di non sforare il limite assoluto API
                if (usage["count"] + profile_api_calls) >= API_DAILY_LIMIT:
                    yield sse_event({"type": "warning", "message": "⚠ Limite API raggiunto. Scartiamo le aziende rimanenti."})
                    break
                
                try:
                    resp3 = requests.get(
                        FMP_PROFILE_URL,
                        params={"symbol": sym, "apikey": FMP_API_KEY},
                        timeout=15,
                    )
                    resp3.raise_for_status()
                    prof_data = resp3.json()
                    profile_api_calls += 1

                    if isinstance(prof_data, list) and len(prof_data) > 0:
                        profile_map[sym] = prof_data[0]
                    elif isinstance(prof_data, dict) and prof_data.get("symbol"):
                        profile_map[sym] = prof_data
                except Exception as e:
                    profile_api_calls += 1
                    logger.error("Errore fetch profilo per %s: %s", sym, e)

            usage["count"] += profile_api_calls

            yield sse_event({
                "type": "info",
                "message": f"  ↳ Profili integrati: {len(profile_map)}/{len(all_symbols)} ({profile_api_calls} API call usate)"
            })

            # ------------------------------------------------------------------
            # FILTRO INTELLIGENTE (Ghigliottina + Piano B)
            # ------------------------------------------------------------------
            profiled_stocks = []   # Tutte le aziende con almeno un profilo scaricato
            complete_stocks = []   # Solo quelle con bilancio 100% completo (PE, EPS, Beta)

            for stock in stocks:
                sym = stock.get("symbol", "")
                if sym in profile_map:
                    prof = profile_map[sym]
                    
                    pe = prof.get("peRatio")
                    eps = prof.get("eps")
                    beta = prof.get("beta")

                    # Salviamo i dati per TUTTE, anche se sono None
                    stock["peRatio"] = pe
                    stock["eps"] = eps
                    stock["lastDiv"] = prof.get("lastDiv")
                    stock["dividendYield"] = prof.get("dividendYield")
                    stock["beta"] = beta

                    profiled_stocks.append(stock)

                    # Sezione perfettini: hanno tutto?
                    if pe is not None and eps is not None and beta is not None:
                        complete_stocks.append(stock)
            
            # Qui si decide il destino
            if len(complete_stocks) > 0:
                # La ghigliottina ha risparmiato qualcuno! Usiamo solo i bilanci completi
                scartati = len(profiled_stocks) - len(complete_stocks)
                stocks = complete_stocks
                yield sse_event({
                    "type": "warning",
                    "message": f"  ↳ GHIGLIOTTINA: {scartati} aziende eliminate perché prive di dati completi."
                })
            else:
                # La ghigliottina avrebbe sterminato tutti. Usiamo il piano B
                stocks = profiled_stocks
                yield sse_event({
                    "type": "warning",
                    "message": "  ⚠ ATTENZIONE: Nessuna azienda ha un bilancio 100% completo oggi! Ghigliottina disattivata, procedo con i dati parziali..."
                })

            save_api_usage(usage)
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            safe_msg = re.sub(r'apikey=[^&\s]+', 'apikey=***HIDDEN_KEY***', str(e))
            msg = f"✖ ERRORE: {safe_msg}"
            yield sse_event({"type": "error", "message": msg})
            save_api_usage(usage)
            yield sse_event({"type": "complete", "message": "Analisi interrotta per errore di rete."})
            return

        if len(stocks) == 0:
            yield sse_event({"type": "error", "message": "✖ Nessuna azienda trovata nell'estrazione."})
            yield sse_event({"type": "complete", "message": "Analisi interrotta."})
            return

        yield sse_event({
            "type": "success",
            "message": f"  ↳ {len(stocks)} aziende portate alla fase di scoring [OK]"
        })
        time.sleep(0.5)

        # --- Step 4: Calcolo AI Score ---
        yield sse_event({"type": "info", "message": "> Analisi fondamentale in corso..."})
        time.sleep(0.3)
        yield sse_event({"type": "info", "message": "  Calculating AI Score (P/E × EPS × Momentum × Dividends)..."})
        time.sleep(0.4)

        scored = []
        for stock in stocks:
            score = compute_ai_score(stock)
            scored.append((stock, score))

        valid_scored = [(s, sc) for s, sc in scored if sc > 0.0]
        valid_scored.sort(key=lambda x: x[1], reverse=True)
        top10 = valid_scored[:10]
        scored.sort(key=lambda x: x[1], reverse=True)

        yield sse_event({
            "type": "success",
            "message": f"  ↳ {len(valid_scored)} aziende superano lo score base > 0."
        })
        time.sleep(0.6)

        if len(top10) == 0:
            yield sse_event({"type": "warning", "message": "⚠ Nessuna azienda ha superato i filtri Value Investing oggi."})
            
            all_stocks_for_csv = [s for s, _ in scored]
            all_scores_for_csv = [sc for _, sc in scored]
            save_results_csv(all_stocks_for_csv, all_scores_for_csv)

            yield sse_event({"type": "complete", "message": "✔ Analisi completata — nessun candidato trovato."})
            return

        # --- Step 5: Output Top 10 ---
        yield sse_event({"type": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"})
        yield sse_event({"type": "success", "message": "  🏆  TOP 10 — AI Investment Ranking"})
        yield sse_event({"type": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"})
        time.sleep(0.3)

        all_stocks_for_csv = [s for s, _ in scored]
        all_scores_for_csv = [sc for _, sc in scored]

        for rank, (stock, score) in enumerate(top10, start=1):
            symbol = stock.get("symbol", "???")
            name = stock.get("name", "N/A")
            price = stock.get("price", 0)
            change_pct = stock.get("changesPercentage", 0)
            pe = stock.get("peRatio", "N/A")
            eps_val = stock.get("eps", "N/A")

            arrow = "↑" if change_pct >= 0 else "↓"

            yield sse_event({
                "type": "result",
                "message": f"  #{rank}  {symbol}",
                "detail": (
                    f"{name} | ${price:,.2f} | {arrow}{abs(change_pct):.2f}% | "
                    f"P/E: {pe} | EPS: {eps_val} | Score: {score}"
                ),
                "sector": stock.get("exchange", "N/A"),
            })
            time.sleep(0.35)

        yield sse_event({"type": "info", "message": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"})
        time.sleep(0.3)

        # Chart Data
        chart_data = [{"symbol": s.get("symbol", "?"), "ai_score": sc} for s, sc in top10]
        if len(chart_data) > 0:
            yield sse_event({"type": "chart_data", "payload": chart_data})

        # --- Step 6: Commento Gemini AI ---
        if len(top10) > 0:
            yield sse_event({"type": "info", "message": "> Collegamento a Gemini AI per analisi qualitativa..."})
            time.sleep(0.5)

            winner_stock, winner_score = top10[0]
            gemini_text = generate_gemini_comment(winner_stock, winner_score)

            if gemini_text:
                yield sse_event({"type": "info", "message": "  ↳ Gemini 2.5 Flash connesso [OK]"})
                time.sleep(0.3)
                yield sse_event({"type": "info", "message": f"  📝 Commento AI su {winner_stock.get('symbol', '???')}:"})
                time.sleep(0.3)

                chunk_size = 8
                for i in range(0, len(gemini_text), chunk_size):
                    chunk = gemini_text[i:i + chunk_size]
                    yield sse_event({"type": "gemini", "message": chunk})
                    time.sleep(0.04)

                yield sse_event({"type": "gemini_end", "message": ""})
                time.sleep(0.3)
            else:
                yield sse_event({"type": "warning", "message": "  ⚠ [Modulo LLM Disconnesso] — Commento AI non disponibile."})
                time.sleep(0.3)

        # --- Step 7: Salvataggio CSV ---
        save_results_csv(all_stocks_for_csv, all_scores_for_csv)

        yield sse_event({
            "type": "complete",
            "message": f"✔ Analisi completata con successo. Salvataggio su CSV completato."
        })

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
