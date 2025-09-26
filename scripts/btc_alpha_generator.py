"""Automatic formulaic alpha generation using Bitcoin data only."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
import yfinance as yf

from alphagen.data.expression import *  # noqa: F401,F403
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs

from alphagen_crypto import BitcoinData, CryptoDataCalculator
from alphagen_crypto.features import close, high, low, open_, target, volume, vwap  # noqa: F401


LOGGER = logging.getLogger(__name__)
@dataclass
class AlphaResult:
    expression: str
    train_ic: float
    train_ric: float
    test_ic: float
    test_ric: float


@dataclass
class EnsembleResult:
    expressions: List[str]
    weights: List[float]
    train_ic: float
    train_ric: float
    test_ic: float
    test_ric: float


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def _load_csv(path: Path) -> pd.DataFrame:
    LOGGER.info("Loading BTC data from %s", path)
    df = pd.read_csv(path, parse_dates=[0])
    df = df.set_index(df.columns[0])
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV file missing required columns: {sorted(missing)}")
    return df.sort_index()


def _fetch_yfinance(start: str, end: str) -> pd.DataFrame:
    LOGGER.info("Fetching Bitcoin OHLCV data from Yahoo Finance")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # yfinance treats the ``end`` argument as exclusive, so expand it by a day so
    # that the user supplied end date is included in the window we return.
    download_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = yf.download(
        "BTC-USD",
        start=start_ts.strftime("%Y-%m-%d"),
        end=download_end,
        progress=False,
        auto_adjust=False,
        interval="1d",
    )
    if data.empty:
        raise RuntimeError("No data returned from Yahoo Finance for BTC-USD")

    data = data.rename(columns=str.title)
    if "Adj Close" in data.columns:
        data = data.drop(columns=["Adj Close"])

    data.index = data.index.tz_localize(None)
    data = data.sort_index()

    start_filtered = pd.Timestamp(start)
    end_filtered = pd.Timestamp(end)
    window = data.loc[(data.index >= start_filtered) & (data.index <= end_filtered)]
    if window.empty:
        raise ValueError("No Bitcoin data retrieved for the specified window")
    return window


def _prepare_data(
    *,
    start: str,
    end: str,
    split: str,
    csv_path: Optional[Path],
    device: torch.device,
    max_backtrack: int,
    max_future: int,
) -> Tuple[BitcoinData, BitcoinData]:
    if csv_path is not None:
        df = _load_csv(csv_path)
    else:
        df = _fetch_yfinance(start, end)

    split_ts = pd.Timestamp(split)
    train_df = df.loc[df.index <= split_ts]
    test_df = df.loc[df.index > split_ts]
    if len(train_df) < 200:
        raise ValueError("Training split must contain at least 200 observations")
    if len(test_df) < 60:
        LOGGER.warning("Test split has fewer than 60 observations; metrics may be unstable")

    train_data = BitcoinData(
        train_df,
        max_backtrack_days=max_backtrack,
        max_future_days=max_future,
        device=device,
    )
    test_data = BitcoinData(
        test_df,
        max_backtrack_days=max_backtrack,
        max_future_days=max_future,
        device=device,
    )
    return train_data, test_data


def _build_gp(population_size: int, generations: int, seed: int, functions) -> SymbolicRegressor:
    estimator = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        init_depth=(2, 5),
        tournament_size=max(population_size // 2, 2),
        stopping_criteria=1.0,
        p_crossover=0.3,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01,
        p_point_mutation=0.1,
        p_point_replace=0.6,
        max_samples=0.9,
        verbose=1,
        parsimony_coefficient=0.0,
        random_state=seed,
        function_set=functions,
        const_range=None,
        n_jobs=1,
    )
    return estimator


def _evaluate_cache(
    cache: Dict[str, float],
    calculator_train: CryptoDataCalculator,
    calculator_test: CryptoDataCalculator,
    top_n: int,
    device: torch.device,
) -> Tuple[List[AlphaResult], EnsembleResult]:
    ordered = sorted(cache.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    results: List[AlphaResult] = []
    expressions: List[str] = []
    expr_objects = []
    for expr_str, _ in ordered:
        expr = eval(expr_str)
        expressions.append(expr_str)
        expr_objects.append(expr)
        train_ic, train_ric = calculator_train.calc_single_all_ret(expr)
        test_ic, test_ric = calculator_test.calc_single_all_ret(expr)
        results.append(AlphaResult(expr_str, train_ic, train_ric, test_ic, test_ric))

    pool = MseAlphaPool(
        capacity=len(expr_objects),
        calculator=calculator_train,
        ic_lower_bound=None,
        device=device,
    )
    pool.force_load_exprs(expr_objects)
    train_ic, train_ric = pool.test_ensemble(calculator_train)
    test_ic, test_ric = pool.test_ensemble(calculator_test)
    ensemble = EnsembleResult(
        expressions=expressions,
        weights=list(pool.weights),
        train_ic=train_ic,
        train_ric=train_ric,
        test_ic=test_ic,
        test_ric=test_ric,
    )
    return results, ensemble


def generate_alphas(args) -> Dict[str, object]:
    reseed_everything(args.seed)
    device = torch.device(args.device)

    train_data, test_data = _prepare_data(
        start=args.start,
        end=args.end,
        split=args.split,
        csv_path=Path(args.csv_path) if args.csv_path else None,
        device=device,
        max_backtrack=args.max_backtrack,
        max_future=args.max_future,
    )
    calculator_train = CryptoDataCalculator(train_data, target)
    calculator_test = CryptoDataCalculator(test_data, target)

    functions = [make_function(**func._asdict()) for func in generic_funcs]
    cache: Dict[str, float] = {}

    def _metric(y_true, y_pred, sample_weight):
        key = y_pred[0]
        if key in cache:
            return cache[key]
        try:
            expr = eval(key)
            ic = calculator_train.calc_single_IC_ret(expr)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("Expression %s failed during evaluation: %s", key, exc)
            ic = -1.0
        if np.isnan(ic):
            ic = -1.0
        cache[key] = ic
        return ic

    metric = make_fitness(function=_metric, greater_is_better=True)
    estimator = _build_gp(args.population_size, args.generations, args.seed, functions)

    features = ["open_", "close", "high", "low", "volume", "vwap"]
    constants = [f"Constant({val})" for val in [-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01,
                                                 0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]]
    terminals = features + constants
    X_train = np.array([terminals])
    y_train = np.array([[1]])

    estimator.set_params(metric=metric)
    estimator.fit(X_train, y_train)

    results, ensemble = _evaluate_cache(cache, calculator_train, calculator_test, args.top_n, device)
    settings = vars(args).copy()
    return {
        "settings": settings,
        "n_evaluated": len(cache),
        "alphas": [result.__dict__ for result in results],
        "ensemble": {
            "expressions": ensemble.expressions,
            "weights": ensemble.weights,
            "train_ic": ensemble.train_ic,
            "train_ric": ensemble.train_ric,
            "test_ic": ensemble.test_ic,
            "test_ric": ensemble.test_ric,
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2010-07-17", help="Start date for the dataset (inclusive)")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="End date for the dataset")
    parser.add_argument("--split", default="2020-01-01", help="Split date between train/test")
    parser.add_argument("--population-size", type=int, default=500, help="Population size for GP")
    parser.add_argument("--generations", type=int, default=30, help="Number of GP generations")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top alphas to report")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--device", default="cpu", help="Torch device to use")
    parser.add_argument("--csv-path", help="Optional CSV file with Bitcoin OHLCV data")
    parser.add_argument("--max-backtrack", type=int, default=120, help="Maximum historical lookback in days")
    parser.add_argument("--max-future", type=int, default=60, help="Future horizon in days for targets")
    parser.add_argument("--output", help="Optional path to save results as JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _setup_logging(args.verbose)
    results = generate_alphas(args)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        LOGGER.info("Saved results to %s", output_path)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
