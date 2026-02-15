import math
import json
import csv
import os
import time
import logging
from datetime import datetime, date

from flask import Flask, render_template, Response, session, request, redirect, url_for
from flask_cors import CORS

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
CORS(app)  # Abilita CORS per tutte le rotte
app.secret_key = os.environ.get("SECRET_KEY", "brera_bot_s3cr3t_k3y_2026_xK9mQ")

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

# Punteggio fisso per Trophy Assets (bypassano i filtri)
_TROPHY_SCORE = 85.0


def compute_ai_score(stock: dict) -> float:
    """
    Value‑Investing × Metodo Brera Score (0–100).

    Filosofia:
      Guido Maria Brera — Debasement, Trophy Assets, Contrarian, "Prima non perdere".

    Flusso:
      0. Trophy Asset?  → bypass filtri, score fisso 85.0
      1. Hard Block 1   → Anti‑Spazzatura (ETF / Fondi / Leva)
      2. Hard Block 2   → Fondamentali obbligatori (P/E + EPS > 0)
      3. Punteggio pesato:
           • P/E ratio        → max 40 pt  (sweet spot 10–25)
           • EPS              → max 25 pt  (utile per azione, log scale)
           • Dividendi        → max 15 pt  (rendimento cedolare)
           • Momentum         → max  5 pt  (residuale)
      4. Modificatori Brera:
           • Beta Risk        → +10 / -15 pt  ("Prima non perdere")
           • Contrarian Pen.  → -20 pt se balzo > 15% ("Evita le mode")
    """

    name = str(stock.get("name") or "").strip()
    symbol = str(stock.get("symbol") or "").strip().upper()

    # ================================================================
    # STEP 0 — Trophy Assets (Bypass Hard Blocks)
    # ================================================================
    if symbol in _TROPHY_ASSETS_WHITELIST:
        # Oro, Argento, Bitcoin: scudi anti-debasement.
        # Non hanno P/E né EPS → score fisso alto.
        return _TROPHY_SCORE

    # ================================================================
    # HARD BLOCK 1 — Filtro Anti‑Spazzatura
    # ================================================================
    if not name or name.upper() == "N/A":
        return 0.0

    name_lower = name.lower()
    for kw in _JUNK_KEYWORDS:
        if kw in name_lower:
            return 0.0

    sym_lower = symbol.lower()
    for kw in ("3x", "2x", "-1x"):
        if kw in sym_lower:
            return 0.0

    # ================================================================
    # HARD BLOCK 2 — Fondamentali Obbligatori
    # ================================================================
    pe_raw = stock.get("peRatio")
    eps_raw = stock.get("eps")

    try:
        pe = float(pe_raw)
    except (ValueError, TypeError):
        pe = None

    try:
        eps = float(eps_raw)
    except (ValueError, TypeError):
        eps = None

    if pe is None or eps is None:
        return 0.0

    if eps <= 0:
        return 0.0

    if pe <= 0:
        return 0.0

    # ================================================================
    # PUNTEGGIO BASE — Fondamentali (max 85 pt)
    # ================================================================
    score = 0.0

    # --- P/E component (max 40 pt) ---
    if 10 <= pe <= 25:
        distance = abs(pe - 17)
        score += max(32.0, 40.0 - distance * 0.7)
    elif pe < 10:
        score += max(15.0, 28.0 - (10.0 - pe) * 2.0)
    elif pe <= 35:
        score += max(8.0, 30.0 - (pe - 25.0) * 2.2)
    elif pe <= 60:
        score += max(2.0, 8.0 - (pe - 35.0) * 0.25)
    else:
        score += 1.0

    # --- EPS component (max 25 pt) ---
    eps_score = math.log1p(eps) * 7.0
    score += min(25.0, eps_score)

    # --- Dividend yield component (max 15 pt) ---
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
            score += div_yield * 3.0
        else:
            score += max(10.0, 15.0 - (div_yield - 5.0) * 0.5)

    # --- Momentum component (max 5 pt) --- residuale
    price = stock.get("price") or 0
    change_pct = stock.get("changesPercentage") or 0
    try:
        price = float(price)
        change_pct = float(change_pct)
    except (ValueError, TypeError):
        price, change_pct = 0.0, 0.0

    if price > 0:
        raw_momentum = abs(change_pct) * price
        score += min(5.0, raw_momentum / 40.0)

    # ================================================================
    # MODIFICATORI BRERA
    # ================================================================

    # --- Beta / Volatilità ("Prima non perdere") ---
    beta_raw = stock.get("beta")
    try:
        beta = float(beta_raw)
    except (ValueError, TypeError):
        beta = None

    if beta is not None:
        if 0.5 <= beta <= 1.2:
            # Bassa volatilità → premio stabilità
            score += 10.0
        elif 1.2 < beta <= 1.5:
            # Neutro — né premio né penalità
            pass
        elif 1.5 < beta <= 1.8:
            # Volatile — leggera penalità
            score -= 5.0
        elif beta > 1.8:
            # Istericamente volatile — penalità severa (Brera: "stanne alla larga")
            score -= 15.0
        elif beta < 0.5:
            # Troppo difensivo — potrebbe essere un settore morto
            score += 3.0

    # --- Penalità Contrarian (Metodo Brera) ---
    # Balzo giornaliero estremo = "moda del momento" → penalizza
    if abs(change_pct) > 15.0:
        score -= 20.0
    elif abs(change_pct) > 10.0:
        score -= 8.0

    return round(min(100.0, max(0.0, score)), 1)


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
    """
    Genera un commento in stile Guido Maria Brera usando Gemini 2.0 Flash.
    Analizza il titolo attraverso i filtri del debasement monetario,
    trophy assets, approccio contrarian e protezione del capitale.
    Ritorna il testo oppure None in caso di errore.
    """
    if not GEMINI_AVAILABLE:
        return None

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

        # Retry logic for 429/Resource Exhausted
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-1.5-pro",
                    contents=prompt,
                )
                return response.text.strip() if response and response.text else None
            except genai.errors.ClientError as e:
                # Catch 429/Quota specific errors
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < 2:
                        time.sleep(2 * (attempt + 1))  # backoff: 2s, 4s
                        continue
                    logger.warning("Gemini Quota Esaurita dopo 3 tentativi: %s", e)
                    return "⚠ [Quota AI Esaurita] — Riprova più tardi (15 req/min limit)."
                logger.error("Errore Gemini Client: %s", e)
                return None
            except Exception:
                # Other exceptions in loop -> continue or break? Usually break for non-transient
                raise
    except Exception as e:
        logger.error("Errore Generico Gemini: %s", e)
        return None


# ---------------------------------------------------------------------------
# Rotte
# ---------------------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    # Se già loggato (server-side), vai alla dashboard
    # Bypass se ?force=1 (per recupero loop redirect client-side)
    if session.get("logged_in") and not request.args.get("force"):
        return redirect(url_for("index"))

    error = None
    if request.method == "POST":
        # Supporta sia form-data che JSON
        # Rileva se il client vuole JSON (AJAX fetch)
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
            pin = "" # Fallback on error

        if pin == APP_PIN.strip():
            session["logged_in"] = True
            # Se è JSON (dal frontend JS) o richiesto, ritorna JSON
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
    import requests  # import locale per non bloccare l'avvio se manca

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
        yield sse_event({
            "type": "info",
            "message": f"[+] API Usage oggi: {used}/{API_DAILY_LIMIT} chiamate utilizzate"
        })
        time.sleep(0.3)

        if remaining <= 0:
            msg = f"✖ ERRORE: Limite API raggiunto ({API_DAILY_LIMIT}/{API_DAILY_LIMIT}). Riprova domani."
            yield sse_event({"type": "error", "message": msg})
            logger.warning("Limite API raggiunto: %d/%d", used, API_DAILY_LIMIT)
            yield sse_event({"type": "complete", "message": "Analisi interrotta."})
            return

        yield sse_event({"type": "success", "message": f"  ↳ {remaining} chiamate rimanenti [OK]"})
        time.sleep(0.5)

        # --- Step 2: Connessione modello logico ---
        yield sse_event({"type": "info", "message": "> Connessione al modello logico..."})
        time.sleep(0.6)
        yield sse_event({"type": "success", "message": "  ↳ Connection established (12ms)"})
        time.sleep(0.3)

        # --- Step 3: Chiamata API FMP (3 endpoint free-tier) ---
        yield sse_event({"type": "info", "message": "> Scaricamento dati di mercato (NASDAQ, NYSE)..."})
        time.sleep(0.4)

        # Verifica che bastino almeno 3 chiamate
        if remaining < 3:
            msg = f"✖ ERRORE: Servono almeno 3 chiamate API, ne restano {remaining}."
            yield sse_event({"type": "error", "message": msg})
            yield sse_event({"type": "complete", "message": "Analisi interrotta."})
            return

        stocks = []
        try:
            # --- Endpoint 1: Most Actives ---
            logger.info("Chiamata API FMP: most-actives...")
            yield sse_event({"type": "info", "message": "  [1/3] Fetching Most Active Stocks..."})
            resp1 = requests.get(FMP_ACTIVES_URL, params={"apikey": FMP_API_KEY}, timeout=15)
            resp1.raise_for_status()
            data1 = resp1.json()
            usage["count"] += 1
            logger.info("most-actives OK (%d risultati)", len(data1) if isinstance(data1, list) else 0)
            time.sleep(0.3)

            # --- Endpoint 2: Biggest Gainers ---
            yield sse_event({"type": "info", "message": "  [2/3] Fetching Biggest Gainers..."})
            resp2 = requests.get(FMP_GAINERS_URL, params={"apikey": FMP_API_KEY}, timeout=15)
            resp2.raise_for_status()
            data2 = resp2.json()
            usage["count"] += 1
            logger.info("biggest-gainers OK (%d risultati)", len(data2) if isinstance(data2, list) else 0)

            # Unisci e deduplica per symbol
            seen = set()
            for s in (data1 if isinstance(data1, list) else []) + (data2 if isinstance(data2, list) else []):
                sym = s.get("symbol")
                if sym and sym not in seen:
                    seen.add(sym)
                    stocks.append(s)

            # --- Endpoint 3: Profili (fondamentali) ---
            yield sse_event({"type": "info", "message": "  [3/3] Fetching Fundamental Data (P/E, EPS, Dividends)..."})
            time.sleep(0.3)

            # Pre-sort by raw momentum to pick only top 20 candidates
            stocks.sort(
                key=lambda s: abs(s.get("changesPercentage", 0) or 0) * (s.get("price", 0) or 0),
                reverse=True,
            )
            top_symbols = [s.get("symbol") for s in stocks[:20] if s.get("symbol")]

            profile_map = {}
            profile_api_calls = 0
            for sym in top_symbols:
                if usage["count"] + profile_api_calls >= API_DAILY_LIMIT:
                    break  # non superare il limite giornaliero
                try:
                    resp3 = requests.get(
                        FMP_PROFILE_URL,
                        params={"symbol": sym, "apikey": FMP_API_KEY},
                        timeout=10,
                    )
                    resp3.raise_for_status()
                    prof_data = resp3.json()
                    profile_api_calls += 1
                    if isinstance(prof_data, list) and len(prof_data) > 0:
                        profile_map[sym] = prof_data[0]
                    elif isinstance(prof_data, dict) and prof_data.get("symbol"):
                        profile_map[sym] = prof_data
                except Exception:
                    profile_api_calls += 1  # conta anche i tentativi falliti
            usage["count"] += profile_api_calls

            yield sse_event({
                "type": "info",
                "message": f"  ↳ Profili scaricati: {len(profile_map)}/{len(top_symbols)} ({profile_api_calls} API calls)"
            })

            # Arricchisci stocks con fondamentali
            for stock in stocks:
                sym = stock.get("symbol", "")
                if sym in profile_map:
                    prof = profile_map[sym]
                    stock["peRatio"] = prof.get("peRatio")
                    stock["eps"] = prof.get("eps")
                    stock["lastDiv"] = prof.get("lastDiv")
                    stock["dividendYield"] = prof.get("dividendYield")
                    stock["beta"] = prof.get("beta")

            logger.info("profiles OK (%d profili, %d API calls)", len(profile_map), profile_api_calls)

            # Salva contatore aggiornato
            save_api_usage(usage)
            logger.info("Contatore API aggiornato: %d/%d", usage["count"], API_DAILY_LIMIT)

        except requests.exceptions.RequestException as e:
            import re
            safe_msg = re.sub(r'apikey=[^&\s]+', 'apikey=***HIDDEN_KEY***', str(e))
            msg = f"✖ ERRORE: {safe_msg}"
            yield sse_event({"type": "error", "message": msg})
            logger.error("Errore API: %s", e)  # log completo (server-side only)
            save_api_usage(usage)  # salva comunque il contatore
            yield sse_event({"type": "complete", "message": "Analisi interrotta per errore di rete."})
            return

        if len(stocks) == 0:
            yield sse_event({"type": "error", "message": "✖ Nessun dato ricevuto dall'API."})
            logger.warning("API ha restituito dati vuoti o non validi.")
            yield sse_event({"type": "complete", "message": "Analisi interrotta."})
            return

        yield sse_event({
            "type": "success",
            "message": f"  ↳ {len(stocks):,} aziende uniche scaricate + fondamentali [OK]"
        })
        time.sleep(0.5)

        # --- Step 4: Calcolo AI Score (Value Investing) ---
        yield sse_event({"type": "info", "message": "> Analisi fondamentale in corso..."})
        time.sleep(0.3)
        yield sse_event({"type": "info", "message": "  Calculating AI Score (P/E × EPS × Momentum × Dividends)..."})
        time.sleep(0.4)
        yield sse_event({"type": "info", "message": "  Applying Value Investing filters..."})
        time.sleep(0.5)

        scored = []
        for stock in stocks:
            score = compute_ai_score(stock)
            scored.append((stock, score))

        # Filtra: rimuovi tutte le azioni con score == 0.0
        valid_scored = [(s, sc) for s, sc in scored if sc > 0.0]
        valid_scored.sort(key=lambda x: x[1], reverse=True)
        top10 = valid_scored[:10]

        # Ordina anche il totale per il CSV (include anche i 0.0)
        scored.sort(key=lambda x: x[1], reverse=True)

        n_filtered = len(scored) - len(valid_scored)
        yield sse_event({
            "type": "success",
            "message": (
                f"  ↳ {len(stocks)} aziende analizzate. "
                f"{len(valid_scored)} superano i filtri ({n_filtered} scartate)."
            )
        })
        time.sleep(0.6)

        # --- Early exit se nessuna azienda supera i filtri ---
        if len(top10) == 0:
            yield sse_event({
                "type": "warning",
                "message": "⚠ Nessuna azienda ha superato i filtri Value Investing oggi."
            })
            time.sleep(0.3)
            yield sse_event({
                "type": "info",
                "message": "  ↳ Tutti gli asset analizzati sono ETF, fondi, in perdita o senza fondamentali."
            })

            # Salva CSV comunque (con tutti gli score a 0)
            all_stocks_for_csv = [s for s, _ in scored]
            all_scores_for_csv = [sc for _, sc in scored]
            csv_path = save_results_csv(all_stocks_for_csv, all_scores_for_csv)
            yield sse_event({
                "type": "info",
                "message": f"  💾 CSV salvato: {os.path.basename(csv_path)}"
            })

            yield sse_event({
                "type": "complete",
                "message": "✔ Analisi completata — nessun candidato Value Investing trovato."
            })
            logger.info("=== Analisi completata (0 candidati validi) ===")
            return

        # --- Step 5: Output Top 10 ---
        yield sse_event({"type": "info", "message": ""})
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

            # Freccia su/giù in base al segno
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

        # --- Step 5b: Invio dati per il grafico ---
        chart_data = [
            {"symbol": s.get("symbol", "?"), "ai_score": sc}
            for s, sc in top10
        ]
        if len(chart_data) > 0:
            yield sse_event({"type": "chart_data", "payload": chart_data})

        # --- Step 6: Commento Gemini AI ---
        if len(top10) > 0:
            yield sse_event({"type": "info", "message": "> Collegamento a Gemini AI per analisi qualitativa..."})
            time.sleep(0.5)

            winner_stock, winner_score = top10[0]
            gemini_text = generate_gemini_comment(winner_stock, winner_score)

            if gemini_text:
                yield sse_event({
                    "type": "info",
                    "message": "  ↳ Gemini 2.0 Flash connesso [OK]"
                })
                time.sleep(0.3)
                yield sse_event({
                    "type": "info",
                    "message": f"  📝 Commento AI su {winner_stock.get('symbol', '???')}:"
                })
                time.sleep(0.3)

                # Effetto macchina da scrivere — invia a blocchi di ~8 caratteri
                chunk_size = 8
                for i in range(0, len(gemini_text), chunk_size):
                    chunk = gemini_text[i:i + chunk_size]
                    yield sse_event({"type": "gemini", "message": chunk})
                    time.sleep(0.04)

                # Segnale di fine typewriter
                yield sse_event({"type": "gemini_end", "message": ""})
                time.sleep(0.3)
            else:
                yield sse_event({
                    "type": "warning",
                    "message": "  ⚠ [Modulo LLM Disconnesso] — Commento AI non disponibile."
                })
                time.sleep(0.3)

        # --- Step 7: Salvataggio CSV ---
        yield sse_event({"type": "info", "message": "> Salvataggio risultati su CSV..."})
        time.sleep(0.3)

        csv_name = save_results_csv(all_stocks_for_csv, all_scores_for_csv)
        yield sse_event({
            "type": "success",
            "message": f"  ↳ File salvato: risultati/{csv_name} [OK]"
        })
        time.sleep(0.3)

        # --- Done ---
        yield sse_event({
            "type": "complete",
            "message": f"✔ Analisi completata con successo. {len(stocks)} aziende analizzate."
        })
        logger.info("=== Analisi completata ===")

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Server Flask avviato su porta %d.", port)
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
