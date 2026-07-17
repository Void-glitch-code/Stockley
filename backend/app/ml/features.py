"""
Feature engineering for stock price prediction.

Turns a DataFrame of daily OHLCV rows into a feature matrix suitable for
scikit-learn regression: lag features (yesterday's close, 2-days-ago close,
etc.), rolling averages, and simple momentum/volatility signals.

The target (what we're predicting) is next_close: tomorrow's closing price.
"""
import pandas as pd
import numpy as np


LAG_DAYS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20]


def _compute_feature_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Adds all feature columns to df (not the target). Shared by both
    build_features (training) and build_latest_features (live prediction)."""
    df = df.copy().sort_values("date").reset_index(drop=True)

    for lag in LAG_DAYS:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    for window in ROLLING_WINDOWS:
        df[f"close_roll_mean_{window}"] = df["close"].shift(1).rolling(window).mean()
        df[f"close_roll_std_{window}"] = df["close"].shift(1).rolling(window).std()

    df["daily_return"] = df["close"].pct_change()
    df["momentum_5"] = df["close"].shift(1) - df["close"].shift(6)
    df["volume_roll_mean_5"] = df["volume"].shift(1).rolling(5).mean()
    df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]

    feature_cols = (
        [f"close_lag_{lag}" for lag in LAG_DAYS]
        + [f"close_roll_mean_{w}" for w in ROLLING_WINDOWS]
        + [f"close_roll_std_{w}" for w in ROLLING_WINDOWS]
        + ["daily_return", "momentum_5", "volume_roll_mean_5", "high_low_spread"]
    )
    return df, feature_cols


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df must have columns: date, open, high, low, close, volume
    sorted ascending by date (oldest first).

    Returns a new DataFrame with feature columns + 'target' (next day's close)
    and 'target_return' (next day's % change), with rows containing NaNs
    (from lag/rolling warmup and the final row with no next-day target) dropped.
    """
    df, feature_cols = _compute_feature_columns(df)

    # --- Target: next day's close (kept for reference/evaluation) ---
    df["target"] = df["close"].shift(-1)

    # --- Target for training: next day's RETURN, not absolute price ---
    # Tree-based models (RandomForest etc.) cannot extrapolate beyond the
    # range of values seen during training -- if a stock trends up/down,
    # absolute prices in the test period fall outside the training range
    # and the model's predictions get stuck near the training boundary.
    # Returns are far more stable regardless of trend direction, so we
    # train on this instead and reconstruct the price afterward:
    #   predicted_price = today_close * (1 + predicted_return)
    df["target_return"] = (df["close"].shift(-1) / df["close"]) - 1

    result = df[["date", "close"] + feature_cols + ["target", "target_return"]].dropna().reset_index(drop=True)
    return result, feature_cols


def build_latest_features(df: pd.DataFrame) -> pd.Series | None:
    """
    Returns the feature row for the most recent day in df -- i.e. the
    inputs needed to predict TOMORROW's price, using only data up to today.
    Returns None if there isn't enough history yet (rolling windows need
    at least ROLLING_WINDOWS[-1] + 1 prior days).
    """
    df, feature_cols = _compute_feature_columns(df)
    latest_row = df.iloc[-1]

    if latest_row[feature_cols].isna().any():
        return None  # not enough history to compute all rolling/lag features

    return latest_row[["date", "close"] + feature_cols]


if __name__ == "__main__":
    # Quick manual test with fake data, so you can sanity-check this file
    # in isolation before wiring it into the real DB pipeline.
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    fake = pd.DataFrame({
        "date": dates,
        "open": np.linspace(100, 140, 40) + np.random.randn(40),
        "high": np.linspace(101, 141, 40) + np.random.randn(40),
        "low": np.linspace(99, 139, 40) + np.random.randn(40),
        "close": np.linspace(100, 140, 40) + np.random.randn(40),
        "volume": np.random.randint(100000, 500000, 40),
    })
    features, cols = build_features(fake)
    print(f"Feature columns: {cols}")
    print(f"Resulting shape: {features.shape}")
    print(features.head())