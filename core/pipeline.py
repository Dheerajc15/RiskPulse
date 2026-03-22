"""
RiskPulse Core Pipeline
=======================
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from abc import ABC, abstractmethod
from scipy import stats
from arch import arch_model
from typing import List, Optional
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Class 1: MarketDataPipeline
# ============================================================================

class MarketDataPipeline:
    """
    Handles end-to-end market data ingestion, validation, and cleaning.
    
    Parameters
    ----------
    tickers : list of str
        Stock/ETF tickers to pull from Yahoo Finance.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    fred_api_key : str, optional
        API key for FRED data. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html
    fred_series : list of str, optional
        FRED series IDs to pull (e.g., ['DGS10', 'DEXUSEU', 'VIXCLS']).
    """

    def __init__(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        fred_api_key: Optional[str] = None,
        fred_series: Optional[List[str]] = None,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.fred_api_key = fred_api_key
        self.fred_series = fred_series or []
        self.raw_data: Optional[pd.DataFrame] = None
        self.clean_data: Optional[pd.DataFrame] = None
        self._validation_report: dict = {}

    def ingest(self) -> "MarketDataPipeline":
        """
        Pull market data from Yahoo Finance and optionally from FRED.
        Returns self for method chaining.
        """
        logger.info(f"Ingesting data for {self.tickers} from {self.start_date} to {self.end_date}")

        # --- Yahoo Finance ---
        yahoo_data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )

        # Handle single vs multiple tickers
        if len(self.tickers) == 1:
            yahoo_data.columns = pd.MultiIndex.from_product(
                [yahoo_data.columns, self.tickers]
            )

        # Flatten: create columns like 'AAPL_Close', 'AAPL_Volume', etc.
        frames = []
        for ticker in self.tickers:
            ticker_df = yahoo_data.xs(ticker, level=1, axis=1).copy()
            ticker_df.columns = [f"{ticker}_{col}" for col in ticker_df.columns]
            frames.append(ticker_df)

        combined = pd.concat(frames, axis=1)

        # --- FRED Data ---
        if self.fred_api_key and self.fred_series:
            fred = Fred(api_key=self.fred_api_key)
            for series_id in self.fred_series:
                try:
                    fred_data = fred.get_series(
                        series_id,
                        observation_start=self.start_date,
                        observation_end=self.end_date,
                    )
                    combined[f"FRED_{series_id}"] = fred_data
                    logger.info(f"  ✓ FRED series {series_id} loaded")
                except Exception as e:
                    logger.warning(f"  ✗ FRED series {series_id} failed: {e}")

        self.raw_data = combined.copy()
        logger.info(f"Ingestion complete: {combined.shape[0]} rows, {combined.shape[1]} columns")
        return self

    def validate(self, z_threshold: float = 3.0) -> "MarketDataPipeline":
        """
        Run data quality checks:
        - Missing value count per column
        - Z-score outlier detection (flags, does not remove)
        
        Parameters
        ----------
        z_threshold : float
            Z-score threshold for outlier flagging. Default 3.0.
        """
        if self.raw_data is None:
            raise ValueError("No data to validate. Call .ingest() first.")

        df = self.raw_data.copy()
        report = {
            "total_rows": len(df),
            "missing_by_column": df.isnull().sum().to_dict(),
            "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "outliers": {},
        }

        # Z-score outlier detection on numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 10:
                continue
            z_scores = np.abs(stats.zscore(col_data))
            outlier_count = int((z_scores > z_threshold).sum())
            if outlier_count > 0:
                report["outliers"][col] = {
                    "count": outlier_count,
                    "pct": round(outlier_count / len(col_data) * 100, 2),
                }

        self._validation_report = report
        logger.info(f"Validation complete. Outlier columns: {len(report['outliers'])}")
        return self

    def get_validation_report(self) -> dict:
        """Return the validation report from the last .validate() call."""
        return self._validation_report

    def impute(self) -> "MarketDataPipeline":
        """
        Fill missing values using forward-fill followed by linear interpolation.
        Any remaining NaNs at the start are back-filled.
        """
        if self.raw_data is None:
            raise ValueError("No data to impute. Call .ingest() first.")

        df = self.raw_data.copy()
        before_nulls = df.isnull().sum().sum()

        df = df.ffill()                   # Forward fill
        df = df.interpolate(method="linear")  # Linear interpolation
        df = df.bfill()                   # Back fill any leading NaNs

        after_nulls = df.isnull().sum().sum()
        logger.info(f"Imputation: {before_nulls} NaNs → {after_nulls} NaNs")

        self.clean_data = df
        return self

    def to_s3(self, bucket: str, key: str, region: str = "us-east-1") -> str:
        """
        Persist cleaned data to AWS S3 as a CSV.

        Parameters
        ----------
        bucket : str
            S3 bucket name.
        key : str
            S3 object key (path/filename.csv).
        region : str
            AWS region. Default 'us-east-1'.

        Returns
        -------
        str
            The S3 URI of the uploaded object.
        """
        if self.clean_data is None:
            raise ValueError("No clean data to persist. Call .impute() first.")

        import boto3

        s3 = boto3.client("s3", region_name=region)
        csv_buffer = self.clean_data.to_csv()
        s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer)

        s3_uri = f"s3://{bucket}/{key}"
        logger.info(f"Uploaded to {s3_uri}")
        return s3_uri

    def get_clean_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame. Falls back to raw if impute hasn't been called."""
        if self.clean_data is not None:
            return self.clean_data
        if self.raw_data is not None:
            return self.raw_data
        raise ValueError("No data available. Call .ingest() first.")


# ============================================================================
# Class 2: TimeSeriesModel (Abstract Base) + Subclasses
# ============================================================================

class TimeSeriesModel(ABC):
    """
    Abstract base class for time series models.
    All subclasses must implement fit() and predict().
    """

    def __init__(self, data: pd.Series, name: str = "BaseModel"):
        self.data = data
        self.name = name
        self.model = None
        self.fitted = None

    @abstractmethod
    def fit(self):
        """Fit the model to the data."""
        raise NotImplementedError("Subclasses must implement fit()")

    @abstractmethod
    def predict(self, horizon: int = 5):
        """Generate predictions for the given horizon."""
        raise NotImplementedError("Subclasses must implement predict()")


class GARCHModel(TimeSeriesModel):
    """
    GARCH(1,1) volatility model using the arch library.
    
    Parameters
    ----------
    data : pd.Series
        Price series (model computes returns internally).
    p : int
        GARCH lag order for volatility. Default 1.
    q : int
        ARCH lag order for shocks. Default 1.
    """

    def __init__(self, data: pd.Series, p: int = 1, q: int = 1):
        super().__init__(data, name=f"GARCH({p},{q})")
        self.p = p
        self.q = q
        # Compute percentage log returns
        self.returns = np.log(data / data.shift(1)).dropna() * 100

    def fit(self) -> dict:
        """
        Fit GARCH model and return key parameters.

        Returns
        -------
        dict with keys: omega, alpha, beta, persistence, log_likelihood
        """
        am = arch_model(self.returns, vol="Garch", p=self.p, q=self.q, dist="Normal")
        self.fitted = am.fit(disp="off")

        params = self.fitted.params
        result = {
            "omega": round(float(params.get("omega", 0)), 6),
            "alpha": round(float(params.get("alpha[1]", 0)), 6),
            "beta": round(float(params.get("beta[1]", 0)), 6),
            "persistence": round(
                float(params.get("alpha[1]", 0)) + float(params.get("beta[1]", 0)), 6
            ),
            "log_likelihood": round(float(self.fitted.loglikelihood), 2),
        }
        logger.info(f"{self.name} fitted. Persistence: {result['persistence']}")
        return result

    def predict(self, horizon: int = 5) -> pd.DataFrame:
        """
        Forecast conditional variance for the given horizon.

        Parameters
        ----------
        horizon : int
            Number of days ahead to forecast.

        Returns
        -------
        pd.DataFrame with columns ['h.1', 'h.2', ...] representing variance forecasts.
        """
        if self.fitted is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        forecast = self.fitted.forecast(horizon=horizon)
        return forecast.variance.dropna()

    def get_current_volatility(self) -> float:
        """Return the latest annualized conditional volatility."""
        if self.fitted is None:
            raise ValueError("Model not fitted. Call .fit() first.")
        cond_vol = self.fitted.conditional_volatility
        latest_daily_vol = cond_vol.iloc[-1] / 100  # Convert back from percentage
        annualized = latest_daily_vol * np.sqrt(252)
        return round(float(annualized), 6)


class ProphetModel(TimeSeriesModel):
    """
    Facebook Prophet model for trend/seasonality decomposition.
    
    Parameters
    ----------
    data : pd.Series
        Price series with a DatetimeIndex.
    """

    def __init__(self, data: pd.Series):
        super().__init__(data, name="Prophet")

    def fit(self) -> "ProphetModel":
        """Fit Prophet model."""
        from prophet import Prophet

        df = pd.DataFrame({"ds": self.data.index, "y": self.data.values})
        self.model = Prophet(daily_seasonality=False, yearly_seasonality=True)
        self.model.fit(df)
        logger.info("Prophet model fitted.")
        return self

    def predict(self, horizon: int = 30) -> pd.DataFrame:
        """
        Generate forecast for the given horizon (in days).

        Returns
        -------
        pd.DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)


# ============================================================================
# Class 3: RiskEngine
# ============================================================================

class RiskEngine:
    """
    Quantitative risk analytics engine.
    
    Parameters
    ----------
    data : pd.DataFrame
        Cleaned market data DataFrame (output of MarketDataPipeline).
    price_col : str
        Column name containing the price series for risk calculations.
    volume_col : str, optional
        Column name containing volume data (for ADTV calculation).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        price_col: str,
        volume_col: Optional[str] = None,
    ):
        self.data = data
        self.price_col = price_col
        self.volume_col = volume_col
        self.prices = data[price_col].dropna()
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()

    # ----- Phase 1 Methods (basic) -----

    def var(self, confidence: float = 0.95) -> float:
        """
        Historical Value at Risk.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.95 for 95% VaR).

        Returns
        -------
        float
            The VaR as a positive number representing the loss threshold.
        """
        percentile = (1 - confidence) * 100
        var_value = float(np.percentile(self.log_returns, percentile))
        return round(-var_value, 6)  # Convention: VaR is reported as positive

    def cvar(self, confidence: float = 0.95) -> float:
        """
        Conditional Value at Risk (Expected Shortfall).
        Average loss beyond the VaR threshold.
        """
        percentile = (1 - confidence) * 100
        var_threshold = np.percentile(self.log_returns, percentile)
        tail_losses = self.log_returns[self.log_returns <= var_threshold]
        return round(-float(tail_losses.mean()), 6)

    def adtv(self, window: int = 20) -> pd.Series:
        """
        Average Daily Trading Volume — rolling mean of daily volume.
        This is the exact metric JPMC automates for liquidity monitoring.

        Parameters
        ----------
        window : int
            Rolling window in trading days. Default 20 (≈1 month).

        Returns
        -------
        pd.Series
            Rolling ADTV values.
        """
        if self.volume_col is None:
            raise ValueError("volume_col not set. Pass it in the constructor.")
        volume = self.data[self.volume_col].dropna()
        return volume.rolling(window=window).mean()

    def monte_carlo_var(self, n_sims: int = 10000, confidence: float = 0.95) -> float:
        """
        Monte Carlo VaR using geometric Brownian motion.
        
        Simulates n_sims one-day returns from the fitted log-normal distribution
        and takes the (1 - confidence) percentile.
        """
        mu = float(self.log_returns.mean())
        sigma = float(self.log_returns.std())

        np.random.seed(42)
        simulated = np.random.normal(mu, sigma, n_sims)

        percentile = (1 - confidence) * 100
        mc_var = float(np.percentile(simulated, percentile))
        return round(-mc_var, 6)

    # ----- Phase 3 Methods (extended) -----

    def historical_var(self, confidence: float = 0.95) -> dict:
        """
        Detailed historical VaR report.
        
        Returns
        -------
        dict with VaR, CVaR, sample size, and confidence level.
        """
        return {
            "confidence": confidence,
            "var_95": self.var(confidence),
            "cvar_95": self.cvar(confidence),
            "sample_size": len(self.log_returns),
            "mean_daily_return": round(float(self.log_returns.mean()), 6),
            "std_daily_return": round(float(self.log_returns.std()), 6),
        }

    def monte_carlo_var_detailed(
        self, n_sims: int = 10000, horizon: int = 1, confidence: float = 0.95
    ) -> dict:
        """
        Multi-horizon Monte Carlo VaR via log-normal path simulation.
        
        Parameters
        ----------
        n_sims : int
            Number of simulation paths.
        horizon : int
            Number of days to simulate forward.
        confidence : float
            Confidence level.

        Returns
        -------
        dict with mc_var, mc_cvar, simulated distribution stats.
        """
        mu = float(self.log_returns.mean())
        sigma = float(self.log_returns.std())

        np.random.seed(42)
        # Simulate cumulative returns over the horizon
        daily_sims = np.random.normal(mu, sigma, (n_sims, horizon))
        cumulative_returns = daily_sims.sum(axis=1)

        percentile = (1 - confidence) * 100
        mc_var = float(np.percentile(cumulative_returns, percentile))
        tail = cumulative_returns[cumulative_returns <= mc_var]
        mc_cvar = float(tail.mean()) if len(tail) > 0 else mc_var

        return {
            "confidence": confidence,
            "horizon_days": horizon,
            "n_simulations": n_sims,
            "mc_var": round(-mc_var, 6),
            "mc_cvar": round(-mc_cvar, 6),
            "sim_mean": round(float(cumulative_returns.mean()), 6),
            "sim_std": round(float(cumulative_returns.std()), 6),
        }

    def garch_volatility(self, p: int = 1, q: int = 1) -> dict:
        """
        Fit GARCH(p,q) and return volatility metrics.
        Wraps the GARCHModel class from this module.
        
        Returns
        -------
        dict with model params, current annualized vol, and persistence.
        """
        garch = GARCHModel(self.prices, p=p, q=q)
        params = garch.fit()
        current_vol = garch.get_current_volatility()

        return {
            **params,
            "annualized_volatility": current_vol,
            "model": f"GARCH({p},{q})",
        }

    def adtv_report(self, window: int = 20) -> dict:
        """
        ADTV with liquidity flag.
        
        A stock is flagged as 'low_liquidity' if the latest ADTV is below
        the 25th percentile of its own historical ADTV.
        
        Returns
        -------
        dict with current_adtv, mean_adtv, liquidity_flag.
        """
        adtv_series = self.adtv(window=window)
        current = float(adtv_series.iloc[-1])
        mean_adtv = float(adtv_series.mean())
        threshold = float(adtv_series.quantile(0.25))

        return {
            "window": window,
            "current_adtv": round(current, 0),
            "mean_adtv": round(mean_adtv, 0),
            "threshold_25pct": round(threshold, 0),
            "liquidity_flag": "low_liquidity" if current < threshold else "normal",
        }

    def greeks_summary(self) -> str:
        """
        Provide a conceptual summary of Delta and Vega in the context 
        of the instrument being analysed.
        
        Note: This is a documentation method, not a computation. RiskPulse
        analyses equity/FX spot instruments, not options. The Greeks are 
        referenced here for completeness in risk discussions.
        
        Returns
        -------
        str
            Explanatory paragraph on Delta and Vega.
        """
        return (
            f"Greeks Summary for {self.price_col}:\n"
            f"Delta measures the sensitivity of an option's price to a $1 change "
            f"in the underlying asset ({self.price_col.split('_')[0]}). For a spot "
            f"equity position, delta is effectively 1.0 (a $1 move in the stock = "
            f"$1 change in position value). In the context of this risk engine, "
            f"delta-equivalent exposure equals the notional position size.\n\n"
            f"Vega measures sensitivity to a 1-percentage-point change in implied "
            f"volatility. While RiskPulse computes realised (GARCH) volatility "
            f"rather than implied vol, the concept maps directly: if GARCH "
            f"annualised vol for {self.price_col.split('_')[0]} shifts by 1%, "
            f"the expected daily P&L range scales proportionally. For portfolios "
            f"with options overlays, vega becomes critical during earnings events "
            f"and macro shocks (e.g., tariff announcements) when vol can spike "
            f"10–20% intraday."
        )