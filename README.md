# RiskPulse — AI-Augmented Market Risk Intelligence Platform

A production-grade risk analytics system that combines quantitative finance (GARCH, VaR, Monte Carlo), distributed data processing (Dask), and AI-powered market intelligence (RAG, prompt engineering) through a FastAPI interface.

Built as a demonstration of end-to-end financial engineering — from raw market data ingestion to real-time risk API endpoints.

---

## What This System Does

RiskPulse ingests market data from Yahoo Finance and FRED, runs it through a cleaning and validation pipeline, computes institutional-grade risk metrics (VaR, CVaR, GARCH volatility, ADTV), and exposes everything through a REST API. A RAG pipeline adds natural language market intelligence over tariff policy documents and FRED releases.

**Why it exists:** To demonstrate the exact skills required for quantitative technology roles at institutions like JPMC — OOP design, risk math, API development, and applied AI.

---

## Architecture

```
DATA SOURCES                    CORE ENGINE                     OUTPUT LAYER
┌─────────────┐           ┌─────────────────────┐          ┌──────────────────┐
│ Yahoo Finance│──────────▶│  MarketDataPipeline │          │   FastAPI App    │
│ FRED API     │           │  ├─ .ingest()       │          │   ├─ /volatility │
│ AWS S3 / CSV │           │  ├─ .validate()     │─────────▶│   ├─ /var        │
└─────────────┘           │  ├─ .impute()       │          │   ├─ /adtv       │
                          │  └─ .to_s3()        │          │   └─ /rag/query  │
                          └─────────────────────┘          └──────────────────┘
                                     │                              ▲
                          ┌─────────────────────┐          ┌───────┴──────────┐
                          │    RiskEngine        │          │  RAG Pipeline    │
                          │  ├─ .var()           │          │  ├─ ChromaDB     │
                          │  ├─ .cvar()          │          │  ├─ Embeddings   │
                          │  ├─ .monte_carlo_var()│         │  └─ Retrieval    │
                          │  ├─ .garch_vol()     │          └──────────────────┘
                          │  └─ .adtv()          │
                          └─────────────────────┘
```

---

## Class Diagram

```
┌─────────────────────────────────────────────────────┐
│                MarketDataPipeline                    │
├─────────────────────────────────────────────────────┤
│ - tickers: List[str]                                │
│ - start_date: str                                   │
│ - end_date: str                                     │
│ - raw_data: Optional[DataFrame]                     │
│ - clean_data: Optional[DataFrame]                   │
├─────────────────────────────────────────────────────┤
│ + ingest() → self                                   │
│ + validate(z_threshold=3.0) → self                  │
│ + impute() → self                                   │
│ + to_s3(bucket, key) → str                          │
│ + get_clean_data() → DataFrame                      │
└─────────────────────────────────────────────────────┘

┌──────────────────────────────┐
│  TimeSeriesModel (ABC)       │
├──────────────────────────────┤
│ - data: Series               │
│ - name: str                  │
├──────────────────────────────┤
│ + fit() → abstract           │
│ + predict(horizon) → abstract│
└──────────┬───────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌────────────┐ ┌──────────────┐
│ GARCHModel │ │ ProphetModel │
├────────────┤ ├──────────────┤
│ + fit()    │ │ + fit()      │
│ + predict()│ │ + predict()  │
│ + get_     │ └──────────────┘
│   current_ │
│   vol()    │
└────────────┘

┌─────────────────────────────────────────────────────┐
│                   RiskEngine                         │
├─────────────────────────────────────────────────────┤
│ - data: DataFrame                                   │
│ - price_col: str                                    │
│ - volume_col: Optional[str]                         │
│ - log_returns: Series                               │
├─────────────────────────────────────────────────────┤
│ + var(confidence=0.95) → float                      │
│ + cvar(confidence=0.95) → float                     │
│ + monte_carlo_var(n_sims, confidence) → float       │
│ + historical_var(confidence) → dict                 │
│ + monte_carlo_var_detailed(n_sims, horizon) → dict  │
│ + garch_volatility(p=1, q=1) → dict                │
│ + adtv(window=20) → Series                          │
│ + adtv_report(window=20) → dict                     │
│ + greeks_summary() → str                            │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Dheerajc15/RiskPulse.git
cd RiskPulse
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for interactive Swagger UI.

### Run Tests

```bash
python tests/test_phase1.py
python tests/test_phase3_risk.py
python core/rag_chain.py
```

---

## API Endpoints & Sample Responses

### `GET /volatility/{ticker}`

Returns GARCH(1,1) volatility analysis.

```json
{
  "ticker": "UBER",
  "period": {"start": "2019-01-01", "end": "2024-12-31"},
  "garch": {
    "omega": 0.032145,
    "alpha": 0.081234,
    "beta": 0.903456,
    "persistence": 0.98469,
    "log_likelihood": -4521.37,
    "annualized_volatility": 0.4213,
    "model": "GARCH(1,1)"
  }
}
```

### `GET /var/{ticker}`

Returns VaR95, CVaR95, and Monte Carlo VaR.

```json
{
  "ticker": "UBER",
  "period": {"start": "2019-01-01", "end": "2024-12-31"},
  "historical_var": {
    "confidence": 0.95,
    "var_95": 0.042371,
    "cvar_95": 0.063892,
    "sample_size": 1500,
    "mean_daily_return": 0.000512,
    "std_daily_return": 0.028943
  },
  "monte_carlo_var": {
    "confidence": 0.95,
    "horizon_days": 1,
    "n_simulations": 10000,
    "mc_var": 0.047123,
    "mc_cvar": 0.058234,
    "sim_mean": 0.000489,
    "sim_std": 0.028912
  }
}
```

### `GET /adtv/{ticker}`

Returns 20-day ADTV with liquidity flag.

```json
{
  "ticker": "UBER",
  "period": {"start": "2019-01-01", "end": "2024-12-31"},
  "adtv": {
    "window": 20,
    "current_adtv": 24531200,
    "mean_adtv": 28453100,
    "threshold_25pct": 18234500,
    "liquidity_flag": "normal"
  }
}
```

### `POST /rag/query`

RAG-based market intelligence.

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the market impact of Section 301 tariffs on EUR/USD?"}'
```

```json
{
  "question": "What was the market impact of Section 301 tariffs on EUR/USD?",
  "answer": "Based on 3 retrieved document chunks: [Source: section_301_tariffs.txt] EUR/USD dropped from 1.1690 to 1.1550 in the week following the announcement, a 1.2% decline...",
  "sources": [
    {"source": "section_301_tariffs.txt", "relevance_distance": 0.3421}
  ]
}
```

---

## Statistical Methodology

### GARCH (Generalized Autoregressive Conditional Heteroskedasticity)

GARCH models capture **volatility clustering** — the empirical observation that large price moves tend to follow large price moves. The GARCH(1,1) model defines conditional variance as:

```
σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)
```

- **ω (omega):** Long-run variance baseline
- **α (alpha):** Reaction to yesterday's shock. High α = volatility responds quickly to new information
- **β (beta):** Persistence of past volatility. High β = volatility decays slowly
- **Persistence (α + β):** Values near 1.0 indicate volatility shocks are long-lived. A persistence of 0.985 means a shock's half-life is ~46 trading days

RiskPulse uses the `arch` library with maximum likelihood estimation and normal distribution assumption.

### Value at Risk (VaR)

VaR answers: "What is the maximum expected loss over one day at 95% confidence?"

**Historical VaR:** Sort all observed daily returns, take the 5th percentile. No distributional assumptions — purely empirical. Weakness: assumes the past distribution represents future risk.

**Monte Carlo VaR:** Simulate 10,000 return paths from a fitted log-normal distribution (calibrated to observed μ and σ), then take the 5th percentile of simulated outcomes. Strength: can model non-linear payoffs and fat tails.

### CVaR (Conditional Value at Risk / Expected Shortfall)

CVaR answers: "If we DO breach the VaR threshold, how bad is the average loss?"

It is the mean of all returns worse than the VaR cutoff. Regulators (Basel III) prefer CVaR over VaR because VaR tells you the boundary of the worst 5%, while CVaR tells you what happens inside that tail.

### ADTV (Average Daily Trading Volume)

ADTV is the 20-day rolling mean of daily share volume. It is the primary liquidity metric used by institutional desks:

- **Why 20 days?** ≈ 1 trading month. Smooths out single-day spikes while remaining responsive
- **Liquidity flag:** If current ADTV falls below the 25th percentile of its own history, execution risk increases — larger orders will move the market
- **JPMC context:** ADTV monitoring is automated across all covered instruments. A drop in ADTV triggers review of position sizing limits

### Monte Carlo Simulation

The Monte Carlo engine simulates future price paths using geometric Brownian motion:

```
S(t+1) = S(t) · exp((μ - σ²/2)·dt + σ·√dt·Z)
```

Where Z ~ N(0,1). With 10,000 simulations, the 5th percentile of terminal values gives Monte Carlo VaR. The key advantage over historical VaR: you can extend the horizon (5-day, 10-day VaR) without needing overlapping historical windows.

---

## Project Structure

```
RiskPulse/
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI application (4 endpoints)
├── core/
│   ├── __init__.py
│   ├── pipeline.py              # MarketDataPipeline, TimeSeriesModel, RiskEngine
│   ├── dask_pipeline.py         # Dask-enhanced pipeline + benchmark
│   └── rag_chain.py             # RAG: ingest, embed, retrieve, answer
├── notebooks/
│   ├── 02_dask_benchmark.py     # Pandas vs Dask benchmark
│   ├── 03_rag_market_data.py    # RAG pipeline walkthrough
│   └── 04_prompt_portfolio.py   # 5 structured prompts
├── data/
│   └── documents/               # Tariff docs, FRED releases for RAG
├── tests/
│   ├── __init__.py
│   ├── test_phase1.py           # OOP smoke tests
│   └── test_phase3_risk.py      # Risk engine tests
├── requirements.txt
├── LICENSE
└── README.md
```

---

## License

MIT License — see [LICENSE](LICENSE).