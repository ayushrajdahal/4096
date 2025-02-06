from packaging.version import Version
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

from pydantic import BaseModel


TimeFeature = Callable[[pd.PeriodIndex], np.ndarray]


def _sine_cosine(xs, period: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode values of ``xs`` using sine and cosine transformations.
    Returns tuple of (sine_array, cosine_array)
    """
    sine = np.sin(2 * np.pi * xs / period)
    cosine = np.cos(2 * np.pi * xs / period)
    return sine, cosine


def second_of_minute(index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Second of minute encoded as sine and cosine.
    """
    return _sine_cosine(index.second, period=60)


def minute_of_hour(index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minute of hour encoded as sine and cosine.
    """
    return _sine_cosine(index.minute, period=60)


def hour_of_day(index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hour of day encoded as sine and cosine.
    """
    return _sine_cosine(index.hour, period=24)


def day_of_week(index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Day of week encoded as sine and cosine.
    """
    return _sine_cosine(index.dayofweek, period=7)


def day_of_month(index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Day of month encoded as sine and cosine.
    """
    return _sine_cosine(index.day - 1, period=31)


def day_of_year(index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Day of year encoded as sine and cosine.
    """
    return _sine_cosine(index.dayofyear - 1, period=366)


def month_of_year(index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Month of year encoded as sine and cosine.
    """
    return _sine_cosine(index.month - 1, period=12)


def week_of_year(index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
    """
    Week of year encoded as sine and cosine.
    """
    try:
        week = index.isocalendar().week
    except AttributeError:
        week = index.week

    return _sine_cosine(week - 1, period=53)


class Constant(BaseModel):
    """
    Constant time feature using a predefined value.
    """

    value: float = 0.0

    def __call__(self, index: pd.PeriodIndex) -> Tuple[np.ndarray, np.ndarray]:
        return np.full((2, index.shape[0]), self.value)


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]

    if len(base_freq) >= 2 and base_freq.endswith("S"):
        base_freq = base_freq[:-1]
        if Version(pd.__version__) >= Version("2.2.0"):
            base_freq += "E"

    return base_freq


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    features_by_offsets: Dict[Any, List[TimeFeature]] = {
        offsets.YearBegin: [],
        offsets.YearEnd: [],
        offsets.QuarterBegin: [month_of_year],
        offsets.QuarterEnd: [month_of_year],
        offsets.MonthBegin: [month_of_year],
        offsets.MonthEnd: [month_of_year],
        offsets.Week: [day_of_month, week_of_year],
        offsets.Day: [day_of_week, day_of_month, day_of_year],
        offsets.BusinessDay: [day_of_week, day_of_month, day_of_year],
        offsets.Hour: [hour_of_day, day_of_week, day_of_month, day_of_year],
        offsets.Minute: [
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
        offsets.Second: [
            second_of_minute,
            minute_of_hour,
            hour_of_day,
            day_of_week,
            day_of_month,
            day_of_year,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, features in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return features

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}

    The following frequencies are supported:
    
    """

    for offset_cls in features_by_offsets:
        offset = offset_cls()
        supported_freq_msg += (
            f"\t{offset.freqstr.split('-')[0]} - {offset_cls.__name__}"
        )

    raise RuntimeError(supported_freq_msg)

def custom_time_features(dates, freq='h') -> np.ndarray:
    """
    Convert datetime index to cyclic time features.
    Returns array of shape [n_features*2, sequence_length]
    """
    index = pd.PeriodIndex(dates, freq=freq)
    features = []
    
    for fun in time_features_from_frequency_str(freq):
        sine, cosine = fun(index)
        features.extend([sine, cosine])
        
    return np.vstack(features)