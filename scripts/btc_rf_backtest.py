"""Backtest a Random Forest Bitcoin strategy with leverage optimization."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

LOGGER = logging.getLogger(__name__)

def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-07-17", help="Start date for the dataset (inclusive)")
    parser.add_argument(
        "--end",
        default=date.today().strftime("%Y-%m-%d"),
        help="End date for the dataset (inclusive)",
    )
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
    parser.add_argument("--disable-alpha-mining", action="store_true", help="Skip mining formulaic alphas before training the model")
    parser.add_argument("--alpha-population-size", type=int, default=200, help="Population size for the symbolic regression alpha search")
    parser.add_argument("--alpha-generations", type=int, default=12, help="Number of generations for the symbolic regression alpha search")
    parser.add_argument("--alpha-top-n", type=int, default=5, help="Number of mined alphas to append as features")
    parser.add_argument("--alpha-max-backtrack", type=int, default=120, help="Maximum historical lookback used when mining alphas")
    parser.add_argument("--alpha-max-future", type=int, default=60, help="Future horizon used when mining alphas")
    parser.add_argument("--alpha-device", default="cpu", help="Torch device to use during alpha mining")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    LOGGER.info("Fetching Bitcoin data from Yahoo Finance")
    from alphagen_crypto.data_fetching import fetch_btc_ohlcv

    data = fetch_btc_ohlcv(args.start, args.end)

    from alphagen_crypto.backtest import (
        AlphaMiningConfig,
        BacktestConfig,
        plot_backtest,
        run_random_forest_backtest,
    )

    alpha_cfg = AlphaMiningConfig(
        enabled=not args.disable_alpha_mining and args.alpha_top_n > 0,
        population_size=args.alpha_population_size,
        generations=args.alpha_generations,
        top_n=args.alpha_top_n,
        max_backtrack=args.alpha_max_backtrack,
        max_future=args.alpha_max_future,
        device=args.alpha_device,
        seed=args.seed,
        verbose=args.verbose,
    )
    config = BacktestConfig(
        split=args.split,
        lookback=args.lookback,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        risk_aversion=args.risk_aversion,
        max_leverage=args.max_leverage,
        seed=args.seed,
        min_train_size=args.min_train_size,
        min_test_size=args.min_test_size,
        alpha_mining=alpha_cfg,
    )

    result, mining_result = run_random_forest_backtest(data, config)

    if mining_result and mining_result.candidates:
        LOGGER.info(
            "Incorporating %d mined alphas into the feature matrix", len(mining_result.candidates)
        )
        LOGGER.info(
            "Alpha mining used %s and evaluated %d expressions",
            mining_result.engine,
            mining_result.evaluated,
        )
        for idx, candidate in enumerate(mining_result.candidates, start=1):
            LOGGER.info(
                "Alpha %d: %s (train IC %.3f, test IC %.3f)",
                idx,
                candidate.expression_str,
                candidate.train_ic,
                candidate.test_ic,
            )

    plot_backtest(result, args.plot_path)
    LOGGER.info("Saved performance plot to %s", args.plot_path)

    LOGGER.info("Final cumulative return (strategy): %.2f%%", result.final_return * 100)
    LOGGER.info("Final cumulative return (buy & hold): %.2f%%", result.buy_and_hold_return * 100)
    LOGGER.info("Annualized Sharpe ratio (strategy): %.2f", result.annualized_sharpe)


if __name__ == "__main__":  # pragma: no cover
    main()
