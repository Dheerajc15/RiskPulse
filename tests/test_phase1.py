"""Quick smoke test for Phase 1 OOP classes."""

import sys
sys.path.insert(0, ".")

from core.pipeline import MarketDataPipeline, GARCHModel, RiskEngine

# --- Test MarketDataPipeline ---
print("=" * 60)
print("Testing MarketDataPipeline")
print("=" * 60)

pipeline = MarketDataPipeline(
    tickers=["UBER", "SPY"],
    start_date="2019-01-01",
    end_date="2024-12-31",
)

pipeline.ingest().validate().impute()
df = pipeline.get_clean_data()
print(f"\nClean data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nValidation Report:")
for k, v in pipeline.get_validation_report().items():
    print(f"  {k}: {v}")

# --- Test GARCHModel ---
print("\n" + "=" * 60)
print("Testing GARCHModel")
print("=" * 60)

garch = GARCHModel(df["UBER_Close"])
params = garch.fit()
print(f"Params: {params}")
print(f"Annualized Vol: {garch.get_current_volatility()}")

forecast = garch.predict(horizon=5)
print(f"5-day forecast:\n{forecast}")

# --- Test RiskEngine ---
print("\n" + "=" * 60)
print("Testing RiskEngine")
print("=" * 60)

engine = RiskEngine(df, price_col="UBER_Close", volume_col="UBER_Volume")
print(f"VaR(95%):       {engine.var()}")
print(f"CVaR(95%):      {engine.cvar()}")
print(f"MC VaR(95%):    {engine.monte_carlo_var()}")
print(f"ADTV (latest):  {engine.adtv().iloc[-1]:,.0f}")
print(f"\nHistorical VaR Report: {engine.historical_var()}")
print(f"\nMonte Carlo Detailed: {engine.monte_carlo_var_detailed(horizon=5)}")
print(f"\nGARCH Volatility: {engine.garch_volatility()}")
print(f"\nADTV Report: {engine.adtv_report()}")
print(f"\nGreeks Summary:\n{engine.greeks_summary()}")

print("\n✅ Phase 1 — All tests passed!")