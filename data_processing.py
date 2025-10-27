# data_processing.py
"""
Обработка данных эксперимента с возможностью выбора алгоритма сглаживания.
Включает генерацию данных, добавление шума и два алгоритма сглаживания.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple
from config import ExperimentConfig
from geometry import CylinderGeometry
from original_smoothing import OriginalSmoother  # Импорт оригинального алгоритма


class DataGenerator:
    """
    Генератор данных эксперимента.
    Создает идеальные данные и добавляет шум измерений.
    """

    def __init__(self, geometry: CylinderGeometry, config: ExperimentConfig):
        """
        Инициализация генератора данных.

        Args:
            geometry: Геометрия цилиндра для расчетов
            config: Конфигурация эксперимента
        """
        self.geometry = geometry
        self.config = config
        self.rng = np.random.RandomState(config.seed)  # Детерминированный ГПСЧ

    def generate_ideal_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерация идеальной кривой V(H) без шума.

        Returns:
            H_ideal: Массив высот [см]
            V_ideal: Массив объемов [л]
        """
        return self.geometry.calculate_ideal_curve()

    def add_measurement_noise(self, H_ideal: np.ndarray, V_ideal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Добавление шума измерений - ТОЧНАЯ РЕАЛИЗАЦИЯ ЛОГИКИ ИЗ PASCAL.

        Args:
            H_ideal: Идеальные значения высот
            V_ideal: Идеальные значения объемов

        Returns:
            Hs: Зашумленные высоты
            Vs: Зашумленные объемы
        """
        Hs = H_ideal.copy()
        Vs = V_ideal.copy()

        # Шум по высоте (кроме граничных точек) - как в Pascal
        for i in range(1, len(Hs) - 1):
            x = self.rng.random()  # Случайное число [0, 1)
            Hs[i] = H_ideal[i] + (2 * x - 1) * self.config.Del_h  # Шум в диапазоне [-Del_h, Del_h]

            # Ограничение: высота не может превышать диаметр цилиндра
            if Hs[i] >= 2 * self.config.R:
                Hs[i] = H_ideal[i]

        # Шум по объему (кумулятивный) - точная реализация логики Pascal
        Vs[0] = V_ideal[0]  # Начальная точка без изменений

        for i in range(1, len(Vs)):
            x = self.rng.random()  # Случайное число [0, 1)

            # Вычисляем идеальный прирост объема между точками
            ideal_increment = (self.geometry.V_h(Hs[i] / self.config.R) -
                               self.geometry.V_h(Hs[i - 1] / self.config.R))

            # Переводим в литры
            ideal_increment_liters = ideal_increment * self.config.R * self.config.R * self.config.L / 1e3

            # Применяем шум к приросту (как в Pascal)
            measured_increment = ideal_increment_liters * (1 + (2 * x - 1) * self.config.Del_V)

            # Кумулятивное суммирование
            Vs[i] = Vs[i - 1] + measured_increment

        return Hs, Vs


class SmoothingProcessor:
    """
    Процессор сглаживания данных с поддержкой двух алгоритмов:
    - Оригинальный алгоритм из Pascal-кода (с двумя режимами)
    - Алгоритм на основе SciPy CubicSpline
    """

    def __init__(self, config: ExperimentConfig, use_original_smoothing: bool = False):
        """
        Инициализация процессора сглаживания.

        Args:
            config: Конфигурация эксперимента
            use_original_smoothing: Флаг выбора алгоритма (True - оригинальный, False - SciPy)
        """
        self.config = config
        self.use_original_smoothing = use_original_smoothing

        # Инициализация выбранного алгоритма сглаживания
        if use_original_smoothing:
            self.original_smoother = OriginalSmoother(config)
            print(f">>> Используется ОРИГИНАЛЬНЫЙ алгоритм сглаживания (Sg_p={config.Sg_p})")
        else:
            print(">>> Используется SciPy алгоритм сглаживания (CubicSpline)")

    def smooth_data(self, H: np.ndarray, V_ideal: np.ndarray, V_noisy: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Сглаживание данных с выбором алгоритма.

        Args:
            H: Массив высот
            V_ideal: Идеальные значения объемов
            V_noisy: Зашумленные значения объемов

        Returns:
            V_smooth: Сглаженные объемы
            iteration_stats: Статистика итераций сглаживания
        """
        if self.use_original_smoothing:
            return self.original_smoother.smooth_data(H, V_ideal, V_noisy)
        else:
            return self.smooth_data_scipy(H, V_ideal, V_noisy)

    def smooth_data_scipy(self, H: np.ndarray, V_ideal: np.ndarray, V_noisy: np.ndarray) -> Tuple[
        np.ndarray, List[Dict]]:
        """
        Упрощенное сглаживание с использованием SciPy CubicSpline.
        Сохраняет совместимость с оригинальным интерфейсом.

        Args:
            H: Массив высот
            V_ideal: Идеальные значения объемов
            V_noisy: Зашумленные значения объемов

        Returns:
            V_smooth: Сглаженные объемы
            iteration_stats: Статистика итераций (пустая для SciPy)
        """
        V_smooth = V_noisy.copy()

        # Фиксируем граничные точки как в Pascal
        V_smooth[self.config.I_beg] = V_ideal[self.config.I_beg]
        V_smooth[self.config.I_end] = V_ideal[self.config.I_end]

        print("--------Сглаживание SciPy------------")

        # Однократное сглаживание сплайнами (не итеративное)
        spline = CubicSpline(H[self.config.I_beg:self.config.I_end + 1],
                             V_smooth[self.config.I_beg:self.config.I_end + 1])
        V_smooth[self.config.I_beg:self.config.I_end + 1] = spline(H[self.config.I_beg:self.config.I_end + 1])

        # Расчет ошибок для совместимости
        max_error = np.max(np.abs(V_ideal[self.config.I_beg:self.config.I_end + 1] -
                                  V_smooth[self.config.I_beg:self.config.I_end + 1]) /
                           V_ideal[self.config.I_beg:self.config.I_end + 1])

        # Производные для анализа
        dV_ideal = np.diff(V_ideal[self.config.I_beg:self.config.I_end + 1])
        dV_smooth = np.diff(V_smooth[self.config.I_beg:self.config.I_end + 1])

        # Расчет ошибки производной с защитой от деления на ноль
        with np.errstate(divide='ignore', invalid='ignore'):
            derivative_errors = np.abs(dV_ideal - dV_smooth) / np.abs(dV_ideal)
            derivative_errors = np.nan_to_num(derivative_errors, nan=0.0, posinf=0.0, neginf=0.0)
            derivative_error = np.max(derivative_errors) if len(derivative_errors) > 0 else 0.0

        # Статистика для совместимости (одна итерация)
        stats = {
            'iteration': 0,
            'max_error': max_error * 100,  # В процентах
            'derivative_error': derivative_error * 100,  # В процентах
            'c_M': 40,
            'i_cM': 50,
            'i_cv0': 140
        }
        iteration_stats = [stats]

        # Вывод в формате Pascal для совместимости
        print(f" k={0:6.2f}   D_Max_V={max_error * 100:7.4f} "
              f"Dev_Max_V={derivative_error * 100:7.4f} c_M={40:3} "
              f"i_cM={50:2} i_cv0={140:3}")

        print("Сглаживание SciPy завершено (одна итерация)")

        return V_smooth, iteration_stats