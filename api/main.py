"""
RiskPulse FastAPI Application
==============================
Four endpoints exposing the OOP risk analytics layer.

Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pipeline import MarketDataPipeline, RiskEngine

app = FastAPI(
    title="RiskPulse API",
    description=(
        "AI-Augmented Market Risk Intelligence Platform.\n\n"
        "Endpoints for GARCH volatility, Value at Risk, ADTV liquidity monitoring, "
        "and RAG-based market intelligence queries."
    ),
    version="1.0.0",
    contact={"name": "Dheeraj", "url": "https://github.com/Dheerajc15/RiskPulse"},
)


# ---- Helpers ----

def _build_engine(ticker: str, start: str = "2019-01-01", end: str = "2024-12-31"):
    """
    Instantiate MarketDataPipeline and RiskEngine for a given ticker.
    This is called per-request. In production you'd cache the data.
    """
    try:
        pipeline = MarketDataPipeline(
            tickers=[ticker],
            start_date=start,
            end_date=end,
        )
        pipeline.ingest().validate().impute()
        df = pipeline.get_clean_data()

        price_col = f"{ticker}_Close"
        volume_col = f"{ticker}_Volume"

        if price_col not in df.columns:
            raise ValueError(f"Column {price_col} not found. Available: {list(df.columns)}")

        engine = RiskEngine(
            df,
            price_col=price_col,
            volume_col=volume_col if volume_col in df.columns else None,
        )
        return engine
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---- Endpoints ----

@app.get("/")
def root():
    """Health check and API info."""
    return {
        "service": "RiskPulse API",
        "version": "1.0.0",
        "endpoints": [
            "/volatility/{ticker}",
            "/var/{ticker}",
            "/adtv/{ticker}",
            "/rag/query",
        ],
        "docs": "/docs",
    }


@app.get("/volatility/{ticker}")
def get_volatility(ticker: str, start: str = "2019-01-01", end: str = "2024-12-31"):
    """
    GARCH volatility analysis for a given ticker.
    
    Returns GARCH(1,1) parameters, persistence, and annualized volatility.
    
    **Example:** GET /volatility/UBER
    """
    engine = _build_engine(ticker.upper(), start, end)
    result = engine.garch_volatility()
    return {
        "ticker": ticker.upper(),
        "period": {"start": start, "end": end},
        "garch": result,
    }


@app.get("/var/{ticker}")
def get_var(
    ticker: str,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    confidence: float = 0.95,
    mc_sims: int = 10000,
    mc_horizon: int = 1,
):
    """
    Value at Risk analysis for a given ticker.
    
    Returns historical VaR, CVaR, and Monte Carlo VaR.
    
    **Example:** GET /var/UBER?confidence=0.95&mc_sims=10000
    """
    engine = _build_engine(ticker.upper(), start, end)
    hist = engine.historical_var(confidence=confidence)
    mc = engine.monte_carlo_var_detailed(n_sims=mc_sims, horizon=mc_horizon, confidence=confidence)

    return {
        "ticker": ticker.upper(),
        "period": {"start": start, "end": end},
        "historical_var": hist,
        "monte_carlo_var": mc,
    }


@app.get("/adtv/{ticker}")
def get_adtv(
    ticker: str,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    window: int = 20,
):
    """
    Average Daily Trading Volume with liquidity flag.
    
    ADTV is the exact metric JPMC automates for liquidity monitoring.
    Flags 'low_liquidity' if current ADTV < 25th percentile of historical.
    
    **Example:** GET /adtv/UBER?window=20
    """
    engine = _build_engine(ticker.upper(), start, end)
    result = engine.adtv_report(window=window)
    return {
        "ticker": ticker.upper(),
        "period": {"start": start, "end": end},
        "adtv": result,
    }


# ---- RAG Endpoint ----

class RAGQuery(BaseModel):
    question: str
    top_k: Optional[int] = 3


@app.post("/rag/query")
def rag_query(query: RAGQuery):
    """
    RAG-based market intelligence query.
    
    Retrieves relevant document chunks from the vector store and 
    generates an answer using the retrieval chain.
    
    **Example body:** {"question": "What was the market impact of Section 301 tariffs on EUR/USD?"}
    """
    try:
        # Try to import the RAG chain (built in Phase 5)
        from core.rag_chain import get_rag_answer
        answer = get_rag_answer(query.question, top_k=query.top_k)
        return answer
    except ImportError:
        # RAG not yet configured — return a helpful placeholder
        return {
            "question": query.question,
            "answer": (
                "RAG pipeline not yet configured. Complete Phase 5 to enable this endpoint. "
                "See notebooks/03_rag_market_data.ipynb for setup instructions."
            ),
            "sources": [],
            "status": "rag_not_configured",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")