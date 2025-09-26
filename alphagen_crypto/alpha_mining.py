"""Tools for mining alpha expressions from Bitcoin historical data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor

from alphagen.data.expression import *  # noqa: F401,F403
from alphagen.models.linear_alpha_pool import MseAlphaPool
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs

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


def _build_symbolic_regressor(
    population_size: int,
    generations: int,
    seed: int,
    function_set: Sequence,
    verbose: bool,
) -> SymbolicRegressor:
    return SymbolicRegressor(
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
        verbose=int(verbose),
        parsimony_coefficient=0.0,
        random_state=seed,
        function_set=function_set,
        const_range=None,
        n_jobs=1,
    )


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
    return candidates, ensemble


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
    function_overrides: Optional[Iterable] = None,
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

    if function_overrides is None:
        function_set = [make_function(**func._asdict()) for func in generic_funcs]
    else:
        function_set = [make_function(**func._asdict()) for func in function_overrides]

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

    features = ("open_", "close", "high", "low", "volume", "vwap")
    terminals = list(features) + [f"Constant({val})" for val in constant_values]

    cache: Dict[str, float] = {}
    expr_cache: Dict[str, Expression] = {}

    def _metric(y_true, y_pred, sample_weight):  # pylint: disable=unused-argument
        key = y_pred[0]
        if key in cache:
            return cache[key]
        try:
            expr = eval(key)
            expr_cache[key] = expr
            ic = calculator_train.calc_single_IC_ret(expr)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("Expression %s failed during evaluation: %s", key, exc)
            ic = -1.0
        if np.isnan(ic):
            ic = -1.0
        cache[key] = float(ic)
        return cache[key]

    metric = make_fitness(function=_metric, greater_is_better=True)
    estimator = _build_symbolic_regressor(
        population_size=population_size,
        generations=generations,
        seed=seed,
        function_set=function_set,
        verbose=verbose,
    )

    X_train = np.array([terminals])
    y_train = np.array([[1]])
    estimator.set_params(metric=metric)
    estimator.fit(X_train, y_train)

    candidates, ensemble = _evaluate_top_candidates(
        cache,
        expr_cache,
        calculator_train,
        calculator_test,
        top_n,
        device,
    )

    return AlphaMiningResult(candidates=candidates, ensemble=ensemble, evaluated=len(cache))
