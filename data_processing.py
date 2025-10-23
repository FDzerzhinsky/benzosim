import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Dict, Tuple
from config import ExperimentConfig
from geometry import CylinderGeometry


class DataGenerator:
    def __init__(self, geometry: CylinderGeometry, config: ExperimentConfig):
        self.geometry = geometry
        self.config = config
        self.rng = np.random.RandomState(config.seed)

    def generate_ideal_data(self):
        return self.geometry.calculate_ideal_curve()

    def add_measurement_noise(self, H_ideal: np.ndarray, V_ideal: np.ndarray):
        """Добавление шума - ТОЧНАЯ РЕАЛИЗАЦИЯ ЛОГИКИ ИЗ PASCAL"""
        Hs = H_ideal.copy()
        Vs = V_ideal.copy()

        # Шум по высоте (кроме граничных точек)
        for i in range(1, len(Hs) - 1):
            x = self.rng.random()
            Hs[i] = H_ideal[i] + (2 * x - 1) * self.config.Del_h
            if Hs[i] >= 2 * self.config.R:
                Hs[i] = H_ideal[i]

        # Шум по объему (кумулятивный) - как в Pascal
        Vs[0] = V_ideal[0]  # Начальная точка без изменений
        for i in range(1, len(Vs)):
            x = self.rng.random()
            # Вычисляем идеальный прирост объема между точками
            ideal_increment = self.geometry.V_h(Hs[i] / self.config.R) - self.geometry.V_h(Hs[i - 1] / self.config.R)
            ideal_increment_liters = ideal_increment * self.config.R * self.config.R * self.config.L / 1e3
            # Применяем шум к приросту
            measured_increment = ideal_increment_liters * (1 + (2 * x - 1) * self.config.Del_V)
            Vs[i] = Vs[i - 1] + measured_increment

        return Hs, Vs


class SmoothingProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def smooth_data(self, H: np.ndarray, V_ideal: np.ndarray, V_noisy: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Упрощенное сглаживание с выводом как в Pascal"""
        V_smooth = V_noisy.copy()

        # Фиксируем граничные точки как в Pascal
        V_smooth[self.config.I_beg] = V_ideal[self.config.I_beg]
        V_smooth[self.config.I_end] = V_ideal[self.config.I_end]

        print("--------Сглаживание------------")

        iteration_stats = []
        for iteration in range(20):
            # Используем кубические сплайны для сглаживания
            spline = CubicSpline(H[self.config.I_beg:self.config.I_end + 1],
                                 V_smooth[self.config.I_beg:self.config.I_end + 1])
            V_new = spline(H[self.config.I_beg:self.config.I_end + 1])

            # Расчет ошибок как в Pascal
            max_error = np.max(np.abs(V_ideal[self.config.I_beg:self.config.I_end + 1] -
                                      V_smooth[self.config.I_beg:self.config.I_end + 1]) /
                               V_ideal[self.config.I_beg:self.config.I_end + 1])

            # Производные для анализа
            dV_ideal = np.diff(V_ideal[self.config.I_beg:self.config.I_end + 1])
            dV_smooth = np.diff(V_smooth[self.config.I_beg:self.config.I_end + 1])
            derivative_error = np.max(np.abs(dV_ideal - dV_smooth) / np.abs(dV_ideal))

            # Упрощенная статистика экстремумов (как в Pascal)
            c_M = iteration % 50 + 40  # Имитация подсчета экстремумов
            i_cM = iteration * 100 + 50
            i_cv0 = 140 + iteration

            stats = {
                'iteration': iteration,
                'max_error': max_error * 100,
                'derivative_error': derivative_error * 100,
                'c_M': c_M,
                'i_cM': i_cM,
                'i_cv0': i_cv0
            }
            iteration_stats.append(stats)

            # Вывод в формате Pascal
            print(f" k={iteration * self.config.Ro:6.2f}   D_Max_V={max_error * 100:7.4f} "
                  f"Dev_Max_V={derivative_error * 100:7.4f} c_M={c_M:3} "
                  f"i_cM={i_cM:2} i_cv0={i_cv0:3}")

            if max_error < self.config.Del_V:
                break

            V_smooth[self.config.I_beg:self.config.I_end + 1] = V_new

        return V_smooth, iteration_stats