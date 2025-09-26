"""Backtesting utilities for Bitcoin strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from sklearn.ensemble import RandomForestRegressor
import torch

from . import BitcoinData, CryptoDataCalculator
from .alpha_mining import AlphaCandidate, AlphaMiningResult, mine_btc_alphas


@dataclass
class AlphaMiningConfig:
    """Configuration controlling the optional alpha mining stage."""

    enabled: bool = True
    population_size: int = 200
    generations: int = 12
    top_n: int = 5
    max_backtrack: int = 120
    max_future: int = 60
    device: str = "cpu"
    seed: int = 7
    verbose: bool = False


@dataclass
class BacktestConfig:
    """Parameters for the Random Forest backtest."""

    split: str = "2019-12-31"
    lookback: int = 10
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 3
    risk_aversion: float = 5.0
    max_leverage: float = 3.0
    seed: int = 7
    min_train_size: int = 200
    min_test_size: int = 30
    alpha_mining: AlphaMiningConfig = field(default_factory=AlphaMiningConfig)


@dataclass
class BacktestResult:
    """Outcome of the Random Forest backtest."""

    equity_curve: pd.Series
    buy_and_hold: pd.Series
    leverage: pd.Series
    predictions: pd.Series
    realized_returns: pd.Series
    mined_alphas: Optional[List[str]] = None

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

    def to_dict(self) -> dict:
        return {
            "final_return": self.final_return,
            "buy_and_hold_return": self.buy_and_hold_return,
            "annualized_sharpe": self.annualized_sharpe,
            "n_periods": int(len(self.equity_curve)),
            "mined_alphas": self.mined_alphas or [],
        }


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


def _compute_alpha_feature_frame(
    dataframe: pd.DataFrame,
    candidates: Sequence[AlphaCandidate],
    *,
    device: torch.device,
    max_backtrack: int,
    max_future: int,
) -> pd.DataFrame:
    dataset = BitcoinData(
        dataframe,
        max_backtrack_days=max_backtrack,
        max_future_days=max_future,
        device=device,
    )
    calculator = CryptoDataCalculator(dataset)
    frames: List[pd.DataFrame] = []
    for idx, candidate in enumerate(candidates, start=1):
        tensor = calculator.evaluate_alpha(candidate.expression)
        df = dataset.make_dataframe(tensor, columns=[f"alpha_{idx}"])
        df = df.xs(dataset.symbol, level=1)
        frames.append(df.rename(columns={f"alpha_{idx}": f"alpha_{idx}"}))
    if not frames:
        return pd.DataFrame(index=dataframe.index)
    combined = pd.concat(frames, axis=1)
    combined.index = pd.DatetimeIndex(combined.index)
    combined = combined.sort_index()
    return combined


def _augment_with_mined_alphas(
    dataframe: pd.DataFrame,
    feature_table: pd.DataFrame,
    config: BacktestConfig,
    *,
    mining_result: Optional[AlphaMiningResult],
) -> Tuple[pd.DataFrame, Optional[AlphaMiningResult]]:
    mining_cfg = config.alpha_mining
    if not mining_cfg.enabled:
        return feature_table, mining_result

    if mining_result is None:
        device = torch.device(mining_cfg.device)
        mining_result = mine_btc_alphas(
            dataframe,
            split=config.split,
            population_size=mining_cfg.population_size,
            generations=mining_cfg.generations,
            top_n=mining_cfg.top_n,
            seed=mining_cfg.seed,
            device=device,
            max_backtrack=mining_cfg.max_backtrack,
            max_future=mining_cfg.max_future,
            verbose=mining_cfg.verbose,
        )

    if not mining_result.candidates:
        return feature_table, mining_result

    device = torch.device(mining_cfg.device)
    alpha_frame = _compute_alpha_feature_frame(
        dataframe,
        mining_result.candidates,
        device=device,
        max_backtrack=mining_cfg.max_backtrack,
        max_future=mining_cfg.max_future,
    )
    feature_table = feature_table.join(alpha_frame, how="inner")
    feature_table = feature_table.dropna()
    return feature_table, mining_result


def _fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: BacktestConfig,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def _optimize_leverage(pred_return: float, sigma: float, config: BacktestConfig) -> float:
    sigma2 = max(sigma ** 2, 1e-8)

    def objective(weight: float) -> float:
        utility = weight * pred_return - 0.5 * config.risk_aversion * sigma2 * weight ** 2
        return -utility

    result = optimize.minimize_scalar(
        objective,
        bounds=(-config.max_leverage, config.max_leverage),
        method="bounded",
        options={"xatol": 1e-3},
    )
    if not result.success:
        return float(np.clip(pred_return / (config.risk_aversion * sigma2), -config.max_leverage, config.max_leverage))
    return float(np.clip(result.x, -config.max_leverage, config.max_leverage))


def run_random_forest_backtest(
    dataframe: pd.DataFrame,
    config: BacktestConfig,
    *,
    mining_result: Optional[AlphaMiningResult] = None,
) -> Tuple[BacktestResult, Optional[AlphaMiningResult]]:
    """Run the Random Forest backtest on the supplied dataframe."""

    feature_table = _build_feature_table(dataframe, config.lookback)

    feature_table, mining_result = _augment_with_mined_alphas(
        dataframe,
        feature_table,
        config,
        mining_result=mining_result,
    )

    split_ts = pd.Timestamp(config.split)
    train_mask = feature_table.index <= split_ts
    test_mask = feature_table.index > split_ts

    train_count = int(train_mask.sum())
    test_count = int(test_mask.sum())

    if train_count < config.min_train_size:
        required_idx = config.min_train_size - 1
        if required_idx >= len(feature_table):
            raise ValueError(
                "Dataset does not contain enough rows to satisfy the minimum training size"
            )
        adjusted_split = feature_table.index[required_idx]
        train_mask = feature_table.index <= adjusted_split
        test_mask = feature_table.index > adjusted_split
        train_count = int(train_mask.sum())
        test_count = int(test_mask.sum())

    if test_count < config.min_test_size:
        raise ValueError(
            "Test data must contain at least"
            f" {config.min_test_size} observations after the split"
        )

    train = feature_table.loc[train_mask]
    test = feature_table.loc[test_mask]

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    model = _fit_random_forest(X_train, y_train, config)
    predictions = pd.Series(model.predict(X_test), index=X_test.index, name="prediction")

    sigma = float(y_train.std(ddof=0))
    leverages = predictions.apply(lambda r: _optimize_leverage(float(r), sigma, config))

    strategy_returns = leverages * y_test
    equity_curve = (1 + strategy_returns).cumprod()
    buy_and_hold = (1 + y_test).cumprod()

    mined_alphas = (
        [candidate.expression_str for candidate in mining_result.candidates]
        if mining_result and mining_result.candidates
        else None
    )

    result = BacktestResult(
        equity_curve=equity_curve,
        buy_and_hold=buy_and_hold,
        leverage=leverages,
        predictions=predictions,
        realized_returns=y_test,
        mined_alphas=mined_alphas,
    )
    return result, mining_result


def plot_backtest(result: BacktestResult, output_path: Path) -> None:
    """Save a comparison plot between the strategy and buy-and-hold."""

    import matplotlib.pyplot as plt  # Imported lazily to avoid mandatory dependency

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


__all__ = [
    "AlphaMiningConfig",
    "BacktestConfig",
    "BacktestResult",
    "plot_backtest",
    "run_random_forest_backtest",
]

