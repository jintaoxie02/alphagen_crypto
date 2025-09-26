"""Tensor-based alpha calculator for cryptocurrency data."""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import Tensor

from alphagen.data.calculator import TensorAlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.pytorch_utils import normalize_by_day

from .bitcoin_data import BitcoinData


def _normalize_single_asset(value: Tensor) -> Tensor:
    if value.shape[1] <= 1:
        # For single-asset datasets we skip cross-sectional normalisation to
        # retain signal variance, simply replacing NaNs with zeros.
        return torch.nan_to_num(value, nan=0.0)
    return normalize_by_day(value)


class CryptoDataCalculator(TensorAlphaCalculator):
    """Alpha calculator that works with :class:`BitcoinData`."""

    def __init__(self, data: BitcoinData, target: Optional[Expression] = None) -> None:
        super().__init__(_normalize_single_asset(target.evaluate(data)) if target is not None else None)
        self.data = data

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        return _normalize_single_asset(expr.evaluate(self.data))

    def _flatten(self, tensor: Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().reshape(-1)

    def _corr(self, lhs: Tensor, rhs: Tensor) -> float:
        lhs_np = self._flatten(lhs)
        rhs_np = self._flatten(rhs)
        mask = np.isfinite(lhs_np) & np.isfinite(rhs_np)
        if mask.sum() < 2:
            return 0.0

        lhs_vals = lhs_np[mask]
        rhs_vals = rhs_np[mask]
        lhs_std = float(np.std(lhs_vals))
        rhs_std = float(np.std(rhs_vals))
        if lhs_std <= 1e-12 or rhs_std <= 1e-12:
            return 0.0

        corr = np.corrcoef(lhs_vals, rhs_vals)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(corr)

    def _spearman(self, lhs: Tensor, rhs: Tensor) -> float:
        lhs_np = self._flatten(lhs)
        rhs_np = self._flatten(rhs)
        mask = np.isfinite(lhs_np) & np.isfinite(rhs_np)
        if mask.sum() < 2:
            return 0.0
        corr, _ = spearmanr(lhs_np[mask], rhs_np[mask])
        if np.isnan(corr):
            return 0.0
        return float(corr)

    def calc_single_IC_ret(self, expr: Expression) -> float:
        return self._corr(self.evaluate_alpha(expr), self.target)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return self._spearman(self.evaluate_alpha(expr), self.target)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self.evaluate_alpha(expr)
        return self._corr(value, self.target), self._spearman(value, self.target)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return self._corr(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_pool_all_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float]:
        with torch.no_grad():
            ensemble = self.make_ensemble_alpha(exprs, weights)
            return self._corr(ensemble, self.target), self._spearman(ensemble, self.target)

    def calc_pool_IC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        return self.calc_pool_all_ret(exprs, weights)[0]

    def calc_pool_rIC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        return self.calc_pool_all_ret(exprs, weights)[1]

    @property
    def n_days(self) -> int:
        return self.data.n_days
