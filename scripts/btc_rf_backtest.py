"""Backtest a Random Forest Bitcoin strategy with leverage optimization."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.ensemble import RandomForestRegressor

from alphagen_crypto.data_fetching import fetch_btc_ohlcv

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    buy_and_hold: pd.Series
    leverage: pd.Series
    predictions: pd.Series
    realized_returns: pd.Series

    @property
    def final_return(self) -> float:
        return float(self.equity_curve.iloc[-1] - 1.0)

    @property
    def buy_and_hold_return(self) -> float:
        return float(self.buy_and_hold.iloc[-1] - 1.0)

    @property
    def annualized_sharpe(self) -> float:
        daily_returns = self.equity_curve.pct_change().dropna()
        if daily_returns.empty:
            return float("nan")
        return np.sqrt(252) * daily_returns.mean() / daily_returns.std(ddof=0)


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def _build_feature_table(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.rename(columns=str.lower)
    frame["return"] = frame["close"].pct_change()
    frame["log_return"] = np.log(frame["close"]).diff()
    frame["volume_change"] = frame["volume"].pct_change()
    frame["high_low_spread"] = (frame["high"] - frame["low"]) / frame["close"]
    frame["vwap"] = (frame["high"] + frame["low"] + frame["close"]) / 3

    for window in (3, 7, 14, 30):
        ma_series = frame["close"].rolling(window).mean()
        frame[f"ma_{window}"] = ma_series

        ratio = frame["close"] / frame[f"ma_{window}"]
        if isinstance(ratio, pd.DataFrame):
            ratio = ratio.iloc[:, 0]
        frame[f"ma_ratio_{window}"] = ratio

        frame[f"volatility_{window}"] = frame["return"].rolling(window).std()

    for lag in range(1, lookback + 1):
        frame[f"return_lag_{lag}"] = frame["return"].shift(lag)
        frame[f"volume_lag_{lag}"] = frame["volume_change"].shift(lag)
        frame[f"spread_lag_{lag}"] = frame["high_low_spread"].shift(lag)

    frame["target"] = frame["return"].shift(-1)

    frame = frame.dropna()
    return frame


def _fit_random_forest(X_train: pd.DataFrame, y_train: pd.Series, args: argparse.Namespace) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _optimize_leverage(pred_return: float, sigma: float, args: argparse.Namespace) -> float:
    sigma2 = max(sigma ** 2, 1e-8)

    def objective(weight: float) -> float:
        # Maximize mean-variance utility with SciPy (minimize negative utility).
        utility = weight * pred_return - 0.5 * args.risk_aversion * sigma2 * weight ** 2
        return -utility

    result = optimize.minimize_scalar(
        objective,
        bounds=(-args.max_leverage, args.max_leverage),
        method="bounded",
        options={"xatol": 1e-3},
    )
    if not result.success:
        LOGGER.debug("Leverage optimization failed: %s", result.message)
    return float(np.clip(result.x, -args.max_leverage, args.max_leverage))


def run_backtest(args: argparse.Namespace) -> BacktestResult:
    LOGGER.info("Fetching Bitcoin data from Yahoo Finance")
    data = fetch_btc_ohlcv(args.start, args.end)

    feature_table = _build_feature_table(data, args.lookback)

    split_ts = pd.Timestamp(args.split)
    train_mask = feature_table.index <= split_ts
    test_mask = feature_table.index > split_ts

    train_count = int(train_mask.sum())
    test_count = int(test_mask.sum())

    if train_count < args.min_train_size:
        required_idx = args.min_train_size - 1
        if required_idx >= len(feature_table):
            raise ValueError(
                "Dataset does not contain enough rows to satisfy the minimum"
                f" training size of {args.min_train_size}."
            )
        adjusted_split = feature_table.index[required_idx]
        LOGGER.warning(
            "Requested split %s yields only %d training rows; adjusting split to %s",
            split_ts.date(),
            train_count,
            adjusted_split.date(),
        )
        train_mask = feature_table.index <= adjusted_split
        test_mask = feature_table.index > adjusted_split
        train_count = int(train_mask.sum())
        test_count = int(test_mask.sum())

    if test_count < args.min_test_size:
        raise ValueError(
            "Test data must contain at least"
            f" {args.min_test_size} observations after the split"
        )

    train = feature_table.loc[train_mask]
    test = feature_table.loc[test_mask]

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    model = _fit_random_forest(X_train, y_train, args)
    predictions = pd.Series(model.predict(X_test), index=X_test.index, name="prediction")

    sigma = float(y_train.std(ddof=0))
    leverages = predictions.apply(lambda r: _optimize_leverage(float(r), sigma, args))

    strategy_returns = leverages * y_test
    equity_curve = (1 + strategy_returns).cumprod()
    buy_and_hold = (1 + y_test).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        buy_and_hold=buy_and_hold,
        leverage=leverages,
        predictions=predictions,
        realized_returns=y_test,
    )


def _plot_results(result: BacktestResult, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    result.equity_curve.plot(ax=ax, label="RF Strategy")
    result.buy_and_hold.plot(ax=ax, label="Buy and Hold")
    ax.set_title("Random Forest Bitcoin Strategy vs Buy and Hold")
    ax.set_ylabel("Cumulative Return (Growth of $1)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved performance plot to %s", output_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-07-17", help="Start date for the dataset (inclusive)")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="End date for the dataset (inclusive)")
    parser.add_argument("--split", default="2019-12-31", help="Train/test split date")
    parser.add_argument("--lookback", type=int, default=10, help="Number of lagged days to include in features")
    parser.add_argument("--n-estimators", type=int, default=300, help="Random forest trees")
    parser.add_argument("--max-depth", type=int, default=8, help="Maximum tree depth")
    parser.add_argument("--min-samples-leaf", type=int, default=3, help="Minimum samples per leaf")
    parser.add_argument("--risk-aversion", type=float, default=5.0, help="Risk aversion parameter for leverage optimization")
    parser.add_argument("--max-leverage", type=float, default=3.0, help="Maximum absolute leverage allowed")
    parser.add_argument("--plot-path", type=Path, default=Path("images/btc_rf_strategy.png"), help="Where to save the performance comparison plot")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the random forest")
    parser.add_argument("--min-train-size", type=int, default=200, help="Minimum number of rows required in the training window")
    parser.add_argument("--min-test-size", type=int, default=30, help="Minimum number of rows required in the testing window")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    result = run_backtest(args)
    _plot_results(result, args.plot_path)

    LOGGER.info("Final cumulative return (strategy): %.2f%%", result.final_return * 100)
    LOGGER.info("Final cumulative return (buy & hold): %.2f%%", result.buy_and_hold_return * 100)
    LOGGER.info("Annualized Sharpe ratio (strategy): %.2f", result.annualized_sharpe)


if __name__ == "__main__":  # pragma: no cover
    main()
