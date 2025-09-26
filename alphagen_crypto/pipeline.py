"""End-to-end pipeline for Bitcoin alpha mining and backtesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .alpha_mining import AlphaMiningResult
from .backtest import (
    AlphaMiningConfig,
    BacktestConfig,
    BacktestResult,
    run_random_forest_backtest,
)
from .data_fetching import fetch_btc_ohlcv


@dataclass
class PipelineConfig:
    """Configuration for the combined mining and backtesting pipeline."""

    start: Optional[str] = "2010-07-17"
    end: Optional[str] = None
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass
class PipelineResult:
    """Outputs from the full Bitcoin pipeline."""

    data: pd.DataFrame
    backtest: BacktestResult
    alpha_mining: Optional[AlphaMiningResult]

    def to_dict(self) -> dict:
        start = self.data.index[0] if not self.data.empty else None
        end = self.data.index[-1] if not self.data.empty else None
        return {
            "data": {
                "start": start.strftime("%Y-%m-%d") if start is not None else None,
                "end": end.strftime("%Y-%m-%d") if end is not None else None,
                "rows": int(len(self.data)),
            },
            "backtest": self.backtest.to_dict(),
            "alpha_mining": self.alpha_mining.to_dict() if self.alpha_mining else None,
        }


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full alpha mining and backtest workflow."""

    data = fetch_btc_ohlcv(config.start, config.end)

    mining_result: Optional[AlphaMiningResult] = None
    if config.backtest.alpha_mining.enabled:
        mining_result = None  # run_random_forest_backtest will mine if needed

    backtest_result, mining_result = run_random_forest_backtest(
        data,
        config.backtest,
        mining_result=mining_result,
    )

    return PipelineResult(data=data, backtest=backtest_result, alpha_mining=mining_result)


__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline", "AlphaMiningConfig", "BacktestConfig"]

