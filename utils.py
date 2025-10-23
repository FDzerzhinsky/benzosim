# utils.py
"""
Вспомогательные функции для обработки данных
"""

import numpy as np
from typing import Tuple


def calculate_segment_volume(h: float, R: float) -> float:
    """
    Вычисление объема сегмента круга
    """
    if h <= R:
        alpha = 2 * np.arccos((R - h) / R)
        return R ** 2 / 2 * (alpha - np.sin(alpha))
    else:
        alpha = 2 * np.arccos((h - R) / R)
        return np.pi * R ** 2 - R ** 2 / 2 * (alpha - np.sin(alpha))


def validate_data(H: np.ndarray, V: np.ndarray) -> bool:
    """
    Проверка корректности данных
    """
    if len(H) != len(V):
        return False
    if np.any(H < 0):
        return False
    if np.any(V < 0):
        return False
    if not np.all(np.diff(H) > 0):
        return False
    if not np.all(np.diff(V) >= 0):
        return False

    return True


def calculate_statistics(ideal: np.ndarray, measured: np.ndarray) -> Tuple[float, float, float]:
    """
    Расчет статистик: среднее, мин, макс отклонение
    """
    errors = np.abs(ideal - measured)
    relative_errors = np.abs((ideal - measured) / ideal) * 100

    mean_error = np.mean(errors)
    max_error = np.max(errors)
    max_relative_error = np.max(relative_errors)

    return mean_error, max_error, max_relative_error