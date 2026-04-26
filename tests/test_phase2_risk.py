"""Detailed test for Phase 3 Risk Engine methods."""

import sys
sys.path.insert(0, ".")

from core.pipeline import MarketDataPipeline, RiskEngine
import json

# Ingest data
pipeline = MarketDataPipeline(
    tickers=["UBER"],
    start_date="2019-05-10",  # Uber IPO date
    end_date="2024-12-31",
)
pipeline.ingest().validate().impute()
df = pipeline.get_clean_data()

engine = RiskEngine(df, price_col="UBER_Close", volume_col="UBER_Volume")

# Test each method and print as JSON (same format the API will return)
print("=" * 60)
print("historical_var()")
print("=" * 60)
print(json.dumps(engine.historical_var(), indent=2))

print("\n" + "=" * 60)
print("monte_carlo_var_detailed(n_sims=10000, horizon=5)")
print("=" * 60)
print(json.dumps(engine.monte_carlo_var_detailed(n_sims=10000, horizon=5), indent=2))

print("\n" + "=" * 60)
print("garch_volatility()")
print("=" * 60)
print(json.dumps(engine.garch_volatility(), indent=2))

print("\n" + "=" * 60)
print("adtv_report(window=20)")
print("=" * 60)
print(json.dumps(engine.adtv_report(), indent=2))

print("\n" + "=" * 60)
print("greeks_summary()")
print("=" * 60)
print(engine.greeks_summary())

print("\n✅ Phase 3 — All risk engine tests passed!")