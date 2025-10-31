# utils.py
"""
Вспомогательные классы и функции для обработки данных
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, UnivariateSpline
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os


class DataLoaderXLSX():
    """
    Класс для загрузки данных из XLSX файлов.
    """

    def __init__(self, file_path: str, sheet_names: list, column_names: list):
        self.file_path = file_path
        self.sheet_names = sheet_names
        self.column_names = column_names
        self.data = self.load_data()

    def load_data(self) -> dict:
        data_frames = {}
        for sheet in self.sheet_names:
            df = pd.read_excel(self.file_path, sheet_name=sheet, usecols=self.column_names)
            data_frames[sheet] = df
        return data_frames


def add_measurement_noise(H: np.ndarray, V: np.ndarray, Del_h: float = 0.05, Del_V: float = 0.0015, seed: int = 42) -> \
Tuple[np.ndarray, np.ndarray]:
    """
    ТОЧНОЕ добавление шума как в оригинальном коде.
    """
    rng = np.random.RandomState(seed)

    # Гарантируем вещественный тип для выходных массивов
    Hs = H.astype(np.float64).copy()
    Vs = V.astype(np.float64).copy()

    # Шум по высоте (кроме граничных точек) - ТОЧНО как в Pascal
    for i in range(1, len(Hs) - 1):
        x = rng.random()  # Случайное число [0, 1)
        noise = (2 * x - 1) * Del_h  # Шум в диапазоне [-Del_h, Del_h]
        Hs[i] = H[i] + noise

    # Шум по объему (кумулятивный) - ТОЧНАЯ реализация логики Pascal
    Vs[0] = V[0]  # Начальная точка без изменений

    for i in range(1, len(Vs)):
        x = rng.random()  # Случайное число [0, 1)

        # Идеальный прирост объема между точками
        ideal_increment = V[i] - V[i - 1]

        # Применяем шум к приросту (как в Pascal)
        measured_increment = ideal_increment * (1 + (2 * x - 1) * Del_V)

        # Кумулятивное суммирование
        Vs[i] = Vs[i - 1] + measured_increment

    return Hs, Vs


class OriginalSmoother:
    """
    Оригинальный алгоритм сглаживания из Pascal-кода с улучшенными параметрами.
    """

    def m_spline(self, n: int, i_0: int, i_N: int, gu: int, x: np.ndarray,
                 u: np.ndarray, F0: float, FN: float) -> np.ndarray:
        """
        Точная реализация процедуры M_Spline из Pascal-кода.
        """
        M = np.zeros(len(x))
        h = np.zeros(len(x))
        Alfa = np.zeros(len(x))
        Beta = np.zeros(len(x))

        # Вычисляем шаги между узлами
        for i in range(i_0, i_N):
            if i + 1 < len(x):
                h[i] = x[i + 1] - x[i]

        # Граничные условия
        if gu // 10 == 1:
            Ld0 = 1.0
            dz0 = 6 * ((u[i_0 + 1] - u[i_0]) / h[i_0] - F0) / h[i_0]
        elif gu // 10 == 2:
            Ld0 = 0.0
            dz0 = 2 * F0
        else:
            Ld0 = 0.0
            dz0 = 0.0

        if gu % 10 == 1:
            MuN = 1.0
            dzN = 6 * (FN - (u[i_N] - u[i_N - 1]) / h[i_N - 1]) / h[i_N - 1]
        elif gu % 10 == 2:
            MuN = 0.0
            dzN = 2 * FN
        else:
            MuN = 0.0
            dzN = 0.0

        # Прямая прогонка
        Alfa[i_0 + 1] = -Ld0 / 2
        Beta[i_0 + 1] = dz0 / 2

        for i in range(i_0 + 1, i_N):
            Mui = h[i - 1] / (h[i - 1] + h[i])
            Ldi = 1 - Mui
            Ai = -Mui
            Bi = -Ldi
            Ci = 2.0

            Fi = 6 * ((u[i + 1] - u[i]) / h[i] - (u[i] - u[i - 1]) / h[i - 1]) / (h[i - 1] + h[i])

            denominator = Ci - Alfa[i] * Ai
            if abs(denominator) > 1e-10:
                Alfa[i + 1] = Bi / denominator
                Beta[i + 1] = (Fi + Beta[i] * Ai) / denominator

        # Обратная прогонка
        denominator = 2 + Alfa[i_N] * MuN
        if abs(denominator) > 1e-10:
            M[i_N] = (dzN - Beta[i_N] * MuN) / denominator

        for i in range(i_N - 1, i_0 - 1, -1):
            M[i] = Alfa[i + 1] * M[i + 1] + Beta[i + 1]

        return M

    def smooth_data(self, H: np.ndarray, V_noisy: np.ndarray,
                    I_beg: int, I_end: int, I_p: int = 10,
                    Ro: float = 0.035, Sg_p: int = 1,
                    max_iter: int = 20, target_error: float = 0.001) -> np.ndarray:
        """
        Улучшенная версия оригинального алгоритма сглаживания с параметром target_error.
        """
        n = len(H) - 1
        i_0 = I_beg
        i_n = I_end
        i_k = I_p
        d = Ro

        V_smooth = V_noisy.copy()

        # Фиксируем граничные точки
        V_smooth[i_0] = V_noisy[i_0]
        V_smooth[i_n] = V_noisy[i_n]

        # Вычисляем шаги между узлами
        h = np.zeros(len(H))
        for i in range(i_0 - i_k, i_n + i_k):
            if i + 1 < len(H):
                h[i] = H[i + 1] - H[i]

        # Рабочие массивы
        y0 = V_smooth.copy()
        y1 = V_smooth.copy()

        print(
            f"    Начало сглаживания (Sg_p={Sg_p}): I_p={I_p}, Ro={Ro}, max_iter={max_iter}, target_error={target_error}")

        k = 0
        while k < max_iter:
            k += 1

            if Sg_p == 1:
                # ПЕРВЫЙ РЕЖИМ: метод конечных разностей
                M = np.zeros(len(H))

                for i in range(i_0 - i_k + 1, i_n + i_k):
                    if i < len(M) - 1 and i > 0:
                        term1 = (y0[i + 1] - y0[i]) / h[i]
                        term2 = (y0[i] - y0[i - 1]) / h[i - 1]
                        M[i] = (term1 - term2) / ((h[i - 1] + h[i]) / 2)

                for i in range(i_0 - i_k + 1, i_n + i_k):
                    if i < len(y1):
                        y1[i] = y0[i] + d * M[i]

            elif Sg_p == 2:
                # ВТОРОЙ РЕЖИМ: метод сплайнов
                y0_temp = y1.copy()
                M_spline = self.m_spline(n, i_0 - i_k, i_n + i_k, 22, H, y0_temp, 0.0, 0.0)

                for i in range(i_0 - i_k + 1, i_n + i_k):
                    if i < len(y1) - 1 and i > 0:
                        laplacian = M_spline[i + 1] - 2 * M_spline[i] + M_spline[i - 1]
                        y1[i] = y0_temp[i] - d * laplacian

            y0 = y1.copy()

            # Расчет ошибок с улучшенным условием остановки
            max_error = 0.0
            for i in range(i_0, i_n + 1):
                error_val = abs(V_noisy[i] - y1[i]) / (V_noisy[i] + 1e-10)
                if error_val > max_error:
                    max_error = error_val

            # Улучшенное условие остановки
            if max_error < target_error:
                print(f"    Сглаживание завершено на итерации {k}: достигнута точность {max_error * 100:.4f}%")
                break

            if k == max_iter:
                print(f"    Достигнут лимит итераций {max_iter}. Текущая ошибка: {max_error * 100:.4f}%")

        return y1


def scipy_smooth(H: np.ndarray, V_noisy: np.ndarray,
                 I_beg: int, I_end: int, smooth_factor: float = 0.1) -> np.ndarray:
    """
    Улучшенная версия SciPy сглаживания с параметром smooth_factor.
    Использует UnivariateSpline для регулировки степени сглаживания.
    """
    V_smooth = V_noisy.copy()

    # Фиксируем граничные точки
    V_smooth[I_beg] = V_noisy[I_beg]
    V_smooth[I_end] = V_noisy[I_end]

    # Сглаживание сплайнами с регулируемым параметром сглаживания
    spline = UnivariateSpline(H[I_beg:I_end + 1],
                              V_smooth[I_beg:I_end + 1],
                              s=len(H) * smooth_factor)
    V_smooth[I_beg:I_end + 1] = spline(H[I_beg:I_end + 1])

    print(f"    SciPy сглаживание: smooth_factor={smooth_factor}")

    return V_smooth


def calculate_derivative(H: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Расчет производной dV/dH.
    """
    return np.gradient(V, H)


def save_plot(fig, filename: str):
    """
    Сохранение графика в папку plots/pigl.
    """
    os.makedirs('plots/pigl', exist_ok=True)
    filepath = f'plots/pigl/{filename}'
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"   ✓ График сохранен: {filepath}")