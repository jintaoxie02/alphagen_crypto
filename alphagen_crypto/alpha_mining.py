"""Tools for mining alpha expressions from Bitcoin historical data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from alphagen.data.expression import (  # noqa: F401
    Abs,
    Add,
    Constant,
    Corr,
    Cov,
    Delta,
    Div,
    EMA,
    Expression,
    Log,
    Mad,
    Max,
    Mean,
    Med,
    Min,
    Mul,
    Pow,
    Rank,
    Ref,
    Sign,
    Std,
    Sub,
    Sum,
    Var,
    WMA,
)
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything

from . import BitcoinData, CryptoDataCalculator
from .features import close, high, low, open_, target, volume, vwap  # noqa: F401

LOGGER = logging.getLogger(__name__)


@dataclass
class AlphaCandidate:
    """Summary of an alpha expression discovered during mining."""

    expression: Expression
    expression_str: str
    train_ic: float
    train_ric: float
    test_ic: float
    test_ric: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "expression": self.expression_str,
            "train_ic": self.train_ic,
            "train_ric": self.train_ric,
            "test_ic": self.test_ic,
            "test_ric": self.test_ric,
        }


@dataclass
class AlphaEnsembleSummary:
    expressions: List[str]
    weights: List[float]
    train_ic: float
    train_ric: float
    test_ic: float
    test_ric: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "expressions": self.expressions,
            "weights": self.weights,
            "train_ic": self.train_ic,
            "train_ric": self.train_ric,
            "test_ic": self.test_ic,
            "test_ric": self.test_ric,
        }


@dataclass
class AlphaMiningResult:
    candidates: List[AlphaCandidate]
    ensemble: AlphaEnsembleSummary
    evaluated: int
    engine: str = "alphagen"

    def to_dict(self) -> Dict[str, object]:
        """Serialise the mining summary, including the AlphaGen engine metadata."""
        return {
            "engine": self.engine,
            "evaluated": self.evaluated,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "ensemble": self.ensemble.to_dict(),
        }


def _prepare_datasets(
    dataframe: pd.DataFrame,
    *,
    split: str,
    device: torch.device,
    max_backtrack: int,
    max_future: int,
) -> Tuple[BitcoinData, BitcoinData]:
    dataframe = dataframe.sort_index()
    split_ts = pd.Timestamp(split)
    train_df = dataframe.loc[dataframe.index <= split_ts]
    test_df = dataframe.loc[dataframe.index > split_ts]

    if len(train_df) < 200:
        raise ValueError("Training split must contain at least 200 observations for alpha mining")
    if len(test_df) < 30:
        LOGGER.warning("Test split contains fewer than 30 observations; IC metrics may be noisy")

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


def _evaluate_top_candidates(
    cache: Dict[str, float],
    expressions: Dict[str, Expression],
    calculator_train: CryptoDataCalculator,
    calculator_test: CryptoDataCalculator,
    top_n: int,
    device: torch.device,
) -> Tuple[List[AlphaCandidate], AlphaEnsembleSummary]:
    ordered = sorted(cache.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    candidates: List[AlphaCandidate] = []
    expr_objects: List[Expression] = []
    expr_strings: List[str] = []

    for expr_str, _ in ordered:
        expr = expressions[expr_str]
        expr_objects.append(expr)
        expr_strings.append(expr_str)
        train_ic, train_ric = calculator_train.calc_single_all_ret(expr)
        test_ic, test_ric = calculator_test.calc_single_all_ret(expr)
        candidates.append(
            AlphaCandidate(
                expression=expr,
                expression_str=expr_str,
                train_ic=float(train_ic),
                train_ric=float(train_ric),
                test_ic=float(test_ic),
                test_ric=float(test_ric),
            )
        )

    if expr_objects:
        pool = MseAlphaPool(
            capacity=len(expr_objects),
            calculator=calculator_train,
            ic_lower_bound=None,
            device=device,
        )
        pool.force_load_exprs(expr_objects)
        train_ic, train_ric = pool.test_ensemble(calculator_train)
        test_ic, test_ric = pool.test_ensemble(calculator_test)
        ensemble = AlphaEnsembleSummary(
            expressions=expr_strings,
            weights=list(pool.weights),
            train_ic=float(train_ic),
            train_ric=float(train_ric),
            test_ic=float(test_ic),
            test_ric=float(test_ric),
        )
    else:
        ensemble = AlphaEnsembleSummary(
            expressions=[],
            weights=[],
            train_ic=0.0,
            train_ric=0.0,
            test_ic=0.0,
            test_ric=0.0,
        )
    return candidates, ensemble


def _random_window(rng: np.random.Generator) -> int:
    return int(rng.choice((3, 5, 10, 20, 30, 60, 90)))


def _random_expression(
    rng: np.random.Generator,
    depth: int,
    *,
    constants: Sequence[float],
    require_feature: bool = True,
) -> Expression:
    if depth <= 0:
        feature_choices = (open_, high, low, close, volume, vwap)
        return rng.choice(feature_choices)

    # Bias towards using rolling operators which often provide useful structure.
    op_choices = (
        ["unary"] * 3
        + ["binary"] * 3
        + ["rolling"] * 6
        + ["pair_rolling"] * 2
    )
    op_type = rng.choice(op_choices)

    if op_type == "unary":
        operand = _random_expression(rng, depth - 1, constants=constants, require_feature=require_feature)
        return rng.choice((Abs, Log, Sign))(operand)

    if op_type == "binary":
        left = _random_expression(rng, depth - 1, constants=constants, require_feature=True)
        right_depth = depth - 1 if rng.random() < 0.6 else 0
        if rng.random() < 0.3:
            right = Constant(float(rng.choice(constants)))
        else:
            right = _random_expression(
                rng,
                right_depth,
                constants=constants,
                require_feature=require_feature,
            )
        return rng.choice((Add, Sub, Mul, Div, Pow))(left, right)

    if op_type == "pair_rolling":
        window = _random_window(rng)
        left = _random_expression(rng, depth - 1, constants=constants, require_feature=True)
        right = _random_expression(rng, depth - 1, constants=constants, require_feature=True)
        operator = rng.choice((Cov, Corr))
        return operator(left, right, window)

    # Rolling operator
    window = _random_window(rng)
    operand = _random_expression(rng, depth - 1, constants=constants, require_feature=require_feature)
    operator = rng.choice((
        Mean,
        Sum,
        Std,
        Var,
        Mad,
        Max,
        Min,
        Med,
        Rank,
        Delta,
        WMA,
        EMA,
        Ref,
    ))
    return operator(operand, window)


def _sample_expressions(
    rng: np.random.Generator,
    *,
    total: int,
    max_depth: int,
    constants: Sequence[float],
) -> List[Expression]:
    seen: Dict[str, Expression] = {}
    attempts = 0
    while len(seen) < total and attempts < total * 10:
        expr = _random_expression(rng, rng.integers(1, max_depth + 1), constants=constants)
        expr_str = str(expr)
        if expr_str not in seen:
            seen[expr_str] = expr
        attempts += 1
    return list(seen.values())


def mine_btc_alphas(
    dataframe: pd.DataFrame,
    *,
    split: str,
    population_size: int = 200,
    generations: int = 10,
    top_n: int = 5,
    seed: int = 7,
    device: torch.device = torch.device("cpu"),
    max_backtrack: int = 120,
    max_future: int = 60,
    constant_values: Optional[Sequence[float]] = None,
    verbose: bool = False,
) -> AlphaMiningResult:
    """Run symbolic regression to discover Bitcoin alpha expressions."""

    reseed_everything(seed)
    train_data, test_data = _prepare_datasets(
        dataframe,
        split=split,
        device=device,
        max_backtrack=max_backtrack,
        max_future=max_future,
    )

    calculator_train = CryptoDataCalculator(train_data, target)
    calculator_test = CryptoDataCalculator(test_data, target)

    if constant_values is None:
        constant_values = (
            -30.0,
            -10.0,
            -5.0,
            -2.0,
            -1.0,
            -0.5,
            -0.01,
            0.01,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            30.0,
        )

    total_samples = population_size * max(1, generations)
    max_depth = 4 if generations < 5 else 6

    rng = np.random.default_rng(seed)

    cache: Dict[str, float] = {}
    expr_cache: Dict[str, Expression] = {}

    candidate_expressions = _sample_expressions(
        rng,
        total=total_samples,
        max_depth=max_depth,
        constants=constant_values,
    )

    for expr in candidate_expressions:
        expr_str = str(expr)
        if expr_str in cache:
            continue
        try:
            ic = calculator_train.calc_single_IC_ret(expr)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("Expression %s failed during evaluation: %s", expr_str, exc)
            continue
        if np.isnan(ic):
            continue
        cache[expr_str] = float(ic)
        expr_cache[expr_str] = expr

    if not cache:
        LOGGER.warning("No valid alpha expressions generated; returning empty result")
        return AlphaMiningResult(candidates=[], ensemble=AlphaEnsembleSummary([], [], 0.0, 0.0, 0.0, 0.0), evaluated=0)

    candidates, ensemble = _evaluate_top_candidates(
        cache,
        expr_cache,
        calculator_train,
        calculator_test,
        top_n,
        device,
    )

    return AlphaMiningResult(candidates=candidates, ensemble=ensemble, evaluated=len(cache))
