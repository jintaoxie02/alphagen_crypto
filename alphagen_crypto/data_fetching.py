"""Utilities for retrieving cryptocurrency market data."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_btc_ohlcv(start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Fetch daily Bitcoin OHLCV data from Yahoo Finance.

    Parameters
    ----------
    start:
        Optional start date (inclusive) in ``YYYY-MM-DD`` format.  If ``None`` the
        earliest available observation from Yahoo Finance is used.
    end:
        Optional end date (inclusive) in ``YYYY-MM-DD`` format.  If ``None`` the
        latest available observation is used.

    Returns
    -------
    pandas.DataFrame
        A dataframe indexed by ``DatetimeIndex`` containing the OHLCV columns.

    Raises
    ------
    RuntimeError
        If Yahoo Finance does not return any data for ``BTC-USD``.
    """

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today()

    # yfinance treats ``end`` as exclusive; request the following day so the
    # supplied end date is included in the data that is returned.
    download_end: Optional[str]
    if end_ts is not None:
        download_end = (end_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        download_end = None

    data = yf.download(
        "BTC-USD",
        start=None if start_ts is None else start_ts.strftime("%Y-%m-%d"),
        end=download_end,
        progress=False,
        auto_adjust=False,
        interval="1d",
    )

    if data.empty:
        raise RuntimeError("No data returned from Yahoo Finance for BTC-USD")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(col).title() for col, *_ in data.columns]
    else:
        data = data.rename(columns=str.title)

    if "Adj Close" in data.columns:
        data = data.drop(columns=["Adj Close"])

    data.index = data.index.tz_localize(None)
    data = data.sort_index()

    if start_ts is not None:
        data = data.loc[data.index >= start_ts]
    if end_ts is not None:
        data = data.loc[data.index <= end_ts]

    return data
