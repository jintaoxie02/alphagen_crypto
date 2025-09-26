"""Run the full Bitcoin alpha mining and Random Forest backtest pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence


LOGGER = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-07-17", help="Start date for BTC-USD data (inclusive)")
    parser.add_argument("--end", default=None, help="Optional end date for BTC-USD data (inclusive)")
    parser.add_argument("--split", default="2019-12-31", help="Train/test split date for both alpha mining and backtest")
    parser.add_argument("--lookback", type=int, default=10, help="Number of lagged days to include in engineered features")
    parser.add_argument("--n-estimators", type=int, default=300, help="Random forest tree count")
    parser.add_argument("--max-depth", type=int, default=8, help="Random forest maximum depth")
    parser.add_argument("--min-samples-leaf", type=int, default=3, help="Random forest minimum samples per leaf")
    parser.add_argument("--risk-aversion", type=float, default=5.0, help="Risk aversion parameter for leverage optimization")
    parser.add_argument("--max-leverage", type=float, default=3.0, help="Maximum absolute leverage")
    parser.add_argument("--min-train-size", type=int, default=200, help="Minimum number of training observations")
    parser.add_argument("--min-test-size", type=int, default=30, help="Minimum number of test observations")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for alpha mining and the random forest")
    parser.add_argument("--alpha-population-size", type=int, default=200, help="Population size for symbolic regression")
    parser.add_argument("--alpha-generations", type=int, default=12, help="Number of symbolic regression generations")
    parser.add_argument("--alpha-top-n", type=int, default=5, help="Number of mined alphas to append as features")
    parser.add_argument("--alpha-max-backtrack", type=int, default=120, help="Maximum historical lookback for alpha mining")
    parser.add_argument("--alpha-max-future", type=int, default=60, help="Future horizon for alpha mining targets")
    parser.add_argument("--alpha-device", default="cpu", help="Torch device used when mining alphas")
    parser.add_argument("--disable-alpha-mining", action="store_true", help="Skip the alpha mining stage entirely")
    parser.add_argument("--plot-path", type=Path, default=Path("images/btc_pipeline_strategy.png"), help="Output path for the strategy vs buy-and-hold plot")
    parser.add_argument("--output-json", type=Path, help="Optional path to store a JSON summary of the pipeline run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def _build_configs(args: argparse.Namespace):
    from alphagen_crypto.backtest import AlphaMiningConfig, BacktestConfig
    from alphagen_crypto.pipeline import PipelineConfig

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
    backtest_cfg = BacktestConfig(
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
    return PipelineConfig(start=args.start, end=args.end, backtest=backtest_cfg)


def _log_summary(result) -> None:
    if result.alpha_mining and result.alpha_mining.candidates:
        LOGGER.info(
            "Alpha mining evaluated %d expressions and retained %d candidates",
            result.alpha_mining.evaluated,
            len(result.alpha_mining.candidates),
        )
        for idx, candidate in enumerate(result.alpha_mining.candidates, start=1):
            LOGGER.info(
                "Alpha %d: %s (train IC %.3f, test IC %.3f)",
                idx,
                candidate.expression_str,
                candidate.train_ic,
                candidate.test_ic,
            )
    else:
        LOGGER.info("Alpha mining stage skipped or produced no valid candidates")

    LOGGER.info(
        "Backtest final return: %.2f%% | Buy & Hold: %.2f%% | Sharpe: %.2f",
        result.backtest.final_return * 100,
        result.backtest.buy_and_hold_return * 100,
        result.backtest.annualized_sharpe,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    config = _build_configs(args)
    LOGGER.info("Fetching BTC-USD data from Yahoo Finance")
    from alphagen_crypto.backtest import plot_backtest
    from alphagen_crypto.pipeline import run_pipeline

    result = run_pipeline(config)

    plot_backtest(result.backtest, args.plot_path)
    LOGGER.info("Saved performance plot to %s", args.plot_path)

    _log_summary(result)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as fp:
            json.dump(result.to_dict(), fp, indent=2)
        LOGGER.info("Wrote pipeline summary to %s", args.output_json)


if __name__ == "__main__":  # pragma: no cover
    main()

