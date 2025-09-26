"""Utility dataset wrapper for Bitcoin OHLCV inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .feature_type import CryptoFeatureType


@dataclass
class _PreloadedData:
    tensor: Tensor
    dates: pd.Index
    asset_ids: pd.Index


class BitcoinData:
    """Container mimicking :class:`alphagen_qlib.stock_data.StockData` for Bitcoin."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        symbol: str = "BTC-USD",
        *,
        max_backtrack_days: int = 120,
        max_future_days: int = 30,
        features: Optional[Sequence[CryptoFeatureType]] = None,
        device: torch.device = torch.device("cpu"),
        preloaded: Optional[_PreloadedData] = None,
    ) -> None:
        if preloaded is not None:
            self.data = preloaded.tensor
            self._dates = preloaded.dates
            self._asset_ids = preloaded.asset_ids
            self.max_backtrack_days = max_backtrack_days
            self.max_future_days = max_future_days
            self.device = device
            self._features = list(features) if features is not None else list(CryptoFeatureType)
            self.symbol = symbol
            return

        if dataframe.index.tzinfo is not None or getattr(dataframe.index, "tz", None) is not None:
            dataframe = dataframe.copy()
            dataframe.index = dataframe.index.tz_convert(None)
        dataframe = dataframe.sort_index()

        if dataframe.index.has_duplicates:
            dataframe = dataframe[~dataframe.index.duplicated(keep="first")]

        self.symbol = symbol
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self.device = device
        self._features = list(features) if features is not None else list(CryptoFeatureType)

        feature_arrays = self._build_feature_matrix(dataframe, device)
        n_days = feature_arrays.shape[0]
        n_features = feature_arrays.shape[1]

        padded = torch.full(
            (n_days + max_backtrack_days + max_future_days, n_features, 1),
            float("nan"),
            dtype=feature_arrays.dtype,
            device=device,
        )
        padded[max_backtrack_days:max_backtrack_days + n_days, :, 0] = feature_arrays
        self.data = padded

        freq = self._infer_freq(dataframe.index)
        front_index = pd.date_range(
            end=dataframe.index[0] - freq,
            periods=max_backtrack_days,
            freq=freq,
        ) if max_backtrack_days > 0 else pd.DatetimeIndex([], name=dataframe.index.name)
        back_index = pd.date_range(
            start=dataframe.index[-1] + freq,
            periods=max_future_days,
            freq=freq,
        ) if max_future_days > 0 else pd.DatetimeIndex([], name=dataframe.index.name)
        self._dates = front_index.append(dataframe.index).append(back_index)
        self._asset_ids = pd.Index([symbol])

    @staticmethod
    def _infer_freq(index: pd.Index) -> pd.Timedelta:
        if len(index) < 2:
            return pd.Timedelta(days=1)
        diffs = index.to_series().diff().dropna()
        freq = diffs.median()
        if pd.isna(freq) or freq == pd.Timedelta(0):
            freq = pd.Timedelta(days=1)
        return freq

    def _build_feature_matrix(self, dataframe: pd.DataFrame, device: torch.device) -> Tensor:
        frame = dataframe.copy()
        col_map = {
            CryptoFeatureType.OPEN: "open",
            CryptoFeatureType.CLOSE: "close",
            CryptoFeatureType.HIGH: "high",
            CryptoFeatureType.LOW: "low",
            CryptoFeatureType.VOLUME: "volume",
            CryptoFeatureType.VWAP: "vwap",
        }
        normalized = frame.rename(columns=str.lower)
        if "vwap" not in normalized.columns:
            normalized["vwap"] = (normalized["high"] + normalized["low"] + normalized["close"]) / 3

        missing = [col for col in col_map.values() if col not in normalized.columns]
        if missing:
            raise ValueError(f"Dataframe missing required columns: {missing}")

        ordered_columns = [col_map[feature] for feature in self._features]
        # ``DataFrame.to_numpy`` keeps a consistent shape even when duplicate
        # columns are requested, whereas stacking individual ``Series`` can
        # produce ragged shapes if pandas returns a 2-D array for any column.
        values = normalized.loc[:, ordered_columns].to_numpy(dtype=np.float32)
        return torch.tensor(values, dtype=torch.float32, device=device)

    def __getitem__(self, slc: Union[slice, str]) -> "BitcoinData":
        if isinstance(slc, str):
            return self[self.find_date_slice(slc)]
        if slc.step is not None:
            raise ValueError("Only support slice with step=None")
        start = 0 if slc.start is None else slc.start
        stop = self.n_days if slc.stop is None else slc.stop
        start = max(0, start)
        stop = min(self.n_days, stop)
        total_start = start
        total_stop = stop + self.max_backtrack_days + self.max_future_days
        total_stop = min(total_stop, self.data.shape[0])
        idx_range = slice(total_start, total_stop)
        data = self.data[idx_range]
        return BitcoinData(
            dataframe=pd.DataFrame(),
            symbol=self.symbol,
            max_backtrack_days=self.max_backtrack_days,
            max_future_days=self.max_future_days,
            features=self._features,
            device=self.device,
            preloaded=_PreloadedData(
                tensor=data,
                dates=self._dates[idx_range],
                asset_ids=self._asset_ids,
            ),
        )

    def find_date_index(self, date: str, exclusive: bool = False) -> int:
        ts = pd.Timestamp(date)
        idx = self._dates.searchsorted(ts)
        if exclusive and idx < len(self._dates) and self._dates[idx] == ts:
            idx += 1
        idx -= self.max_backtrack_days
        if idx < 0 or idx > self.n_days:
            raise ValueError(
                f"Date {date} is out of range: available "
                f"[{self._dates[self.max_backtrack_days]}, {self._dates[self.max_backtrack_days + self.n_days - 1]}]"
            )
        return idx

    def find_date_slice(self, start_time: Optional[str] = None, end_time: Optional[str] = None) -> slice:
        start = None if start_time is None else self.find_date_index(start_time)
        stop = None if end_time is None else self.find_date_index(end_time, exclusive=False)
        return slice(start, stop)

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return 1

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    @property
    def stock_ids(self) -> pd.Index:
        return self._asset_ids

    def make_dataframe(
        self,
        data: Union[Tensor, List[Tensor]],
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(
                f"number of days in tensor ({n_days}) doesn't match current data ({self.n_days})"
            )
        if self.n_stocks != n_stocks:
            raise ValueError(
                f"number of assets in tensor ({n_stocks}) doesn't match current data ({self.n_stocks})"
            )
        if len(columns) != n_columns:
            raise ValueError(
                f"size of columns ({len(columns)}) doesn't match tensor feature count ({n_columns})"
            )
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._asset_ids])
        reshaped = data.reshape(-1, n_columns)
        return pd.DataFrame(reshaped.detach().cpu().numpy(), index=index, columns=columns)
