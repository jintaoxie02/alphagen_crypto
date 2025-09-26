"""Utilities for applying AlphaGen to cryptocurrency data."""

from .feature_type import CryptoFeatureType

__all__ = ["CryptoFeatureType", "CryptoDataCalculator", "BitcoinData"]


def __getattr__(name: str):  # pragma: no cover - thin convenience wrapper
    if name == "BitcoinData":
        from .bitcoin_data import BitcoinData as _BitcoinData

        return _BitcoinData
    if name == "CryptoDataCalculator":
        from .calculator import CryptoDataCalculator as _CryptoDataCalculator

        return _CryptoDataCalculator
    raise AttributeError(f"module 'alphagen_crypto' has no attribute {name!r}")
