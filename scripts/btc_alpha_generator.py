"""Automatic formulaic alpha generation using Bitcoin data only."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd
import torch

from alphagen.utils.random import reseed_everything

from alphagen_crypto.alpha_mining import AlphaMiningResult, mine_btc_alphas
from alphagen_crypto.data_fetching import fetch_btc_ohlcv


LOGGER = logging.getLogger(__name__)


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


def generate_alphas(args) -> Dict[str, object]:
    reseed_everything(args.seed)
    device = torch.device(args.device)

    if args.csv_path:
        dataframe = _load_csv(Path(args.csv_path))
    else:
        dataframe = fetch_btc_ohlcv(args.start, args.end)

    mining_result: AlphaMiningResult = mine_btc_alphas(
        dataframe,
        split=args.split,
        population_size=args.population_size,
        generations=args.generations,
        top_n=args.top_n,
        seed=args.seed,
        device=device,
        max_backtrack=args.max_backtrack,
        max_future=args.max_future,
        verbose=args.verbose,
    )

    settings = vars(args).copy()
    return {
        "settings": settings,
        "n_evaluated": mining_result.evaluated,
        "alphas": [candidate.to_dict() for candidate in mining_result.candidates],
        "ensemble": mining_result.ensemble.to_dict(),
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
