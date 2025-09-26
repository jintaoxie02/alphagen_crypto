from enum import IntEnum


class CryptoFeatureType(IntEnum):
    """Feature ordering for cryptocurrency OHLCV data."""

    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5
