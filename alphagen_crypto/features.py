"""Feature shortcuts for cryptocurrency alpha generation."""

from alphagen.data.expression import Feature, Ref

from .feature_type import CryptoFeatureType


high = High = HIGH = Feature(CryptoFeatureType.HIGH)
low = Low = LOW = Feature(CryptoFeatureType.LOW)
volume = Volume = VOLUME = Feature(CryptoFeatureType.VOLUME)
open_ = Open = OPEN = Feature(CryptoFeatureType.OPEN)
close = Close = CLOSE = Feature(CryptoFeatureType.CLOSE)
vwap = Vwap = VWAP = Feature(CryptoFeatureType.VWAP)

# Default target: 20-day forward return.
target = Ref(close, -20) / close - 1
