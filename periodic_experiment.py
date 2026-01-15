# periodic_experiment.py
"""
Эксперимент с периодами опустошения цилиндра.
Наследуется от основного класса CylinderExperiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Any

import pandas as pd
from scipy.interpolate import CubicSpline
import warnings

from experiment import CylinderExperiment
from config import PeriodicExperimentConfig
from geometry import CylinderGeometry
from utils import OriginalSmoother, scipy_smooth, SafeOriginalSmoother, calculate_derivative


class PeriodicDrainingExperiment(CylinderExperiment):
    """
    Эксперимент с периодами опустошения цилиндра.
    Наследует основную функциональность от CylinderExperiment.
    """

    def __init__(self, config: PeriodicExperimentConfig, name: str = ""):
        super().__init__(config, name)
        self.periodic_config = config
        self.periods_data = []  # Список периодов
        self.smoothed_data = {}  # Данные после сглаживания
        self.interpolated_data = {}  # Данные после интерполяции
        self.combined_characteristics = {}  # Объединенные характеристики
        self.combined_Pandas = {}  # Объединенные характеристики в формате Pandas
        self.rng = np.random.RandomState(config.seed)
        self.global_step_counter = 0  # Сквозной счётчик шагов

        # Создаем подкаталог для графиков этого эксперимента
        self.plot_subdir = f"periodic_{name.replace(' ', '_').replace('-', '_')}"
        os.makedirs(f'plots/{self.plot_subdir}', exist_ok=True)

    def run_experiment(self):
        """Переопределенный метод запуска эксперимента с периодами"""
        print("\n" + "=" * 60)
        print(f"ПЕРИОДИЧЕСКИЙ ЭКСПЕРИМЕНТ: {self.name}")
        print("=" * 60)

        # Выводим информацию о используемой конфигурации
        print("ИСПОЛЬЗУЕМАЯ КОНФИГУРАЦИЯ:")
        print(f"  Тип помехи: {self.periodic_config.obstacle_type.value}")
        print(f"  Количество периодов: {self.periodic_config.n_periods}")
        print(
            f"  Диапазон шагов: {self.periodic_config.min_drain_step:.2f}-{self.periodic_config.max_drain_step:.2f} см")
        print(
            f"  Уровни начала: {self.periodic_config.period_start_min:.1f}-{self.periodic_config.period_start_max:.1f} см")
        print(
            f"  Уровни окончания: {self.periodic_config.period_end_min:.1f}-{self.periodic_config.period_end_max:.1f} см")
        print(f"  Точность округления: {self.periodic_config.level_precision} знак(ов)")
        print(f"  Алгоритмы сглаживания: {self.periodic_config.smoothing_algorithms}")
        print(f"  Шаг сетки: {self.periodic_config.grid_step} см")

        try:
            # 1. Инициализация геометрии (как в родительском классе)
            self.config.print_config()
            self.geometry.print_geometry_info()

            # 2. Генерация периодов опустошения
            print(f"\nГЕНЕРАЦИЯ {self.periodic_config.n_periods} ПЕРИОДОВ ОПУСТОШЕНИЯ...")
            self.periods_data = self.generate_periods()

            # 3. Вывод статистики
            self.print_periods_statistics()

            # 4. Визуализация исходных данных (если включено)
            if self.periodic_config.enable_period_plotting:
                print("\nСОЗДАНИЕ ГРАФИКОВ ИСХОДНЫХ ДАННЫХ...")
                self.visualize_periods()

            # 5. Сглаживание данных
            print(f"\nСГЛАЖИВАНИЕ ДАННЫХ...")
            self.smooth_periods_data()
            # Визуализация сглаженных кривых ДО интерполяции/экстраполяции (только периоды)
            try:
                self.visualize_periods_before_extrapolation()
            except Exception as e:
                print(f"Ошибка при visualizing periods before extrapolation: {e}")

            # 6. Интерполяция на сетку
            print(f"\nИНТЕРПОЛЯЦИЯ НА СЕТКУ...")
            self.interpolate_to_grid()

            # 7. Создание объединенной характеристики
            print(f"\nСОЗДАНИЕ ОБЪЕДИНЕННОЙ ХАРАКТЕРИСТИКИ...")
            self.create_combined_characteristics()
            # Сохранить сравнение объединённой характеристики с точной геометрией
            try:
                self.visualize_combined_vs_geometry()
            except Exception as e:
                print(f"Ошибка при visualizing combined vs geometry: {e}")
            # Визуализация производных для объединённых и периодных характеристик
            try:
                self.visualize_derivatives_all()
            except Exception as e:
                print(f"Ошибка при visualizing derivatives: {e}")

            # 8. Визуализация сглаженных данных (если включено)
            if self.periodic_config.enable_smoothing_plots:
                print("\nСОЗДАНИЕ ГРАФИКОВ СГЛАЖЕННЫХ ДАННЫХ...")
                self.visualize_smoothed_curves()

            print(f"\nПЕРИОДИЧЕСКИЙ ЭКСПЕРИМЕНТ ЗАВЕРШЕН!")

            return self.get_experiment_data()

        except Exception as e:
            print(f"Ошибка в периодическом эксперименте: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_periods(self) -> List[Dict]:
        """Генерация всех периодов опустошения"""
        periods = []
        self.global_step_counter = 0

        for period_id in range(self.periodic_config.n_periods):
            print(f"  Генерация периода {period_id + 1}...")
            period = self.generate_single_period(period_id)
            periods.append(period)

        return periods

    def generate_single_period(self, period_id: int) -> Dict:
        """Генерация одного периода опустошения"""
        # Начальный и конечный уровни периода (случайные в заданных диапазонах)
        start_level = self.rng.uniform(
            self.periodic_config.period_start_min,
            self.periodic_config.period_start_max
        )

        end_level = self.rng.uniform(
            self.periodic_config.period_end_min,
            self.periodic_config.period_end_max
        )

        # Гарантируем, что начальный уровень выше конечного
        if start_level <= end_level:
            start_level = end_level + 10.0  # Минимальный зазор 10 см

        points = []

        # Начальные точные значения
        H_ideal_prev = start_level
        V_ideal_prev = self.geometry.config.R * self.geometry.config.R * self.geometry.config.L * \
                       self.geometry.V_h(start_level / self.geometry.config.R) / 1e3
        V_measured_prev = V_ideal_prev  # Начальное значение для кумулятивного шума

        # Шаг 0 - начальная точка периода (с возмущением)
        H_measured = self.round_level(H_ideal_prev)

        # Возмущение начального объема
        x = self.rng.random()
        V_measured = V_ideal_prev * (1 + (2 * x - 1) * self.config.Del_V)

        points.append({
            'step_index': 0,
            'global_step': self.global_step_counter,
            'H_ideal_prev': H_ideal_prev,
            'H_ideal_current': H_ideal_prev,
            'V_ideal_current': V_ideal_prev,
            'H_measured': H_measured,
            'V_measured': V_measured,
            'drain_step': 0.0  # Нет шага опустошения для начальной точки
        })

        self.global_step_counter += 1
        V_measured_prev = V_measured
        step_index = 1

        # Основной цикл опустошения
        while H_ideal_prev > end_level:
            # Случайный шаг опустошения
            drain_step = self.rng.uniform(
                self.periodic_config.min_drain_step,
                self.periodic_config.max_drain_step
            )

            H_ideal_current = H_ideal_prev - drain_step

            # Проверка границ - если вышли за конечный уровень, устанавливаем в end_level
            if H_ideal_current < end_level:
                H_ideal_current = end_level

            # Точный объем от точного уровня
            V_ideal_current = self.geometry.config.R * self.geometry.config.R * self.geometry.config.L * \
                              self.geometry.V_h(H_ideal_current / self.geometry.config.R) / 1e3

            # Округление уровня с заданной точностью
            H_measured = self.round_level(H_ideal_current)

            # Возмущение объема (кумулятивное от предыдущего возмущенного)
            # Приращение вычисляем от точных значений
            ideal_increment = V_ideal_current - V_ideal_prev
            x = self.rng.random()
            measured_increment = ideal_increment * (1 + (2 * x - 1) * self.config.Del_V)
            V_measured = V_measured_prev + measured_increment

            # Сохранение точки
            points.append({
                'step_index': step_index,
                'global_step': self.global_step_counter,
                'H_ideal_prev': H_ideal_prev,
                'H_ideal_current': H_ideal_current,
                'V_ideal_current': V_ideal_current,
                'H_measured': H_measured,
                'V_measured': V_measured,
                'drain_step': drain_step
            })

            # Подготовка к следующему шагу (обновляем ТОЧНЫЕ значения)
            H_ideal_prev = H_ideal_current
            V_ideal_prev = V_ideal_current
            V_measured_prev = V_measured
            step_index += 1
            self.global_step_counter += 1

            # Выход если достигли конечного уровня
            if abs(H_ideal_current - end_level) < 0.001:
                break

        return {
            'period_id': period_id,
            'start_level': start_level,
            'end_level': end_level,
            'points': points
        }

    def round_level(self, level: float) -> float:
        """Округление уровня с заданной точностью"""
        return round(level, self.periodic_config.level_precision)

    def smooth_periods_data(self):
        """Сглаживание данных периодов выбранными алгоритмами"""
        print("Применение алгоритмов сглаживания к периодам...")

        self.smoothed_data = {}

        for algorithm in self.periodic_config.smoothing_algorithms:
            print(f"  Алгоритм: {algorithm}")
            self.smoothed_data[algorithm] = []

            for period in self.periods_data:
                # Извлекаем данные периода
                H_measured = np.array([point['H_measured'] for point in period['points']])
                V_measured = np.array([point['V_measured'] for point in period['points']])

                # В периоде опустошения уровни УБЫВАЮТ, но алгоритмы требуют ВОЗРАСТАНИЯ
                # Разворачиваем массивы для сглаживания
                H_ascending = H_measured[::-1].copy()  # Делаем возрастающим
                V_ascending = V_measured[::-1].copy()

                # Удаляем дубликаты по высоте (они мешают сглаживанию и вызывают деление на ноль).
                # Для одинаковых H усредняем соответствующие V.
                if len(H_ascending) > 0:
                    uniq_H = []
                    uniq_V = []
                    i = 0
                    n = len(H_ascending)
                    while i < n:
                        h_val = H_ascending[i]
                        j = i + 1
                        # поскольку H_ascending отсортирован, дубликаты будут рядом
                        while j < n and np.isclose(H_ascending[j], h_val):
                            j += 1
                        # усредняем V на интервале [i, j)
                        mean_v = float(np.mean(V_ascending[i:j]))
                        uniq_H.append(h_val)
                        uniq_V.append(mean_v)
                        i = j
                    H_ascending_unique = np.array(uniq_H)
                    V_ascending_unique = np.array(uniq_V)
                else:
                    H_ascending_unique = H_ascending
                    V_ascending_unique = V_ascending

                # Применяем выбранный алгоритм сглаживания
                # Если после удаления дублей осталось мало точек, сглаживание бессмысленно — возвращаем исходные значения
                if len(H_ascending_unique) < 2:
                    V_smooth_ascending = V_ascending_unique.copy()
                else:
                    V_smooth_ascending = self.apply_smoothing_algorithm(algorithm, H_ascending_unique, V_ascending_unique)

                # Очистка NaN/inf в результате сглаживания: интерполируем по валидным точкам
                try:
                    V_arr = np.array(V_smooth_ascending, dtype=float)
                    finite_mask = np.isfinite(V_arr)
                    if not np.all(finite_mask):
                        valid_idx = np.where(finite_mask)[0]
                        if valid_idx.size >= 2:
                            # линейная интерполяция по уникальным H
                            V_filled = np.interp(H_ascending_unique, H_ascending_unique[valid_idx], V_arr[valid_idx])
                        elif valid_idx.size == 1:
                            V_filled = np.full_like(V_arr, V_arr[valid_idx[0]], dtype=float)
                        else:
                            # нет валидных значений, возвращаем исходные измеренные
                            V_filled = np.array(V_ascending_unique, dtype=float)
                        V_smooth_ascending = V_filled
                except Exception:
                    # Если что-то пошло не так, оставляем исходные значения
                    V_smooth_ascending = V_ascending_unique.copy()

                # Разворачиваем обратно (это сглаженные значения, соответствующие уникальным H)
                V_smooth = V_smooth_ascending[::-1]

                # Сохраняем сглаженные данные (используем уникальные возрастающие точки для интерполяции)
                smoothed_period = {
                    'period_id': period['period_id'],
                    'H_measured': H_measured,
                    'V_measured': V_measured,
                    'V_smooth': V_smooth,
                    'H_ascending': H_ascending_unique,
                    'V_smooth_ascending': V_smooth_ascending
                }

                self.smoothed_data[algorithm].append(smoothed_period)

    def apply_smoothing_algorithm(self, algorithm: str, H: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Применение конкретного алгоритма сглаживания"""
        try:
            if algorithm == 'scipy':
                # SciPy сглаживание
                I_beg = 0
                I_end = len(H) - 1
                smooth_factor = 0.1
                return scipy_smooth(H, V, I_beg, I_end, smooth_factor)

            elif algorithm == 'original_sg1':
                # Оригинальный алгоритм Sg_p=1 с безопасными параметрами
                smoother = SafeOriginalSmoother()
                safe_params = {
                    'I_p': min(5, len(H) // 4),  # Адаптивный I_p
                    'Ro': 0.01,  # Более консервативный параметр
                    'max_iter': 15,
                    'target_error': 0.005,  # Более реалистичная точность
                    'Sg_p': 1
                }
                return smoother.safe_smooth_data(H, V, **safe_params)

            elif algorithm == 'original_sg2':
                # Оригинальный алгоритм Sg_p=2 с безопасными параметрами
                smoother = SafeOriginalSmoother()
                safe_params = {
                    'I_p': min(5, len(H) // 4),  # Адаптивный I_p
                    'Ro': 0.008,  # Более консервативный параметр
                    'max_iter': 15,
                    'target_error': 0.005,  # Более реалистичная точность
                    'Sg_p': 2
                }
                return smoother.safe_smooth_data(H, V, **safe_params)

            else:
                print(f"  Предупреждение: неизвестный алгоритм {algorithm}, возвращаются исходные данные")
                return V.copy()

        except Exception as e:
            print(f"  Ошибка при сглаживании алгоритмом {algorithm}: {e}")
            print(f"  Возвращаются исходные данные")
            return V.copy()

    def interpolate_to_grid(self):
        """Интерполяция сглаженных данных на равномерную сетку"""
        print("Интерполяция данных на сетку...")

        self.interpolated_data = {}
        grid_step = self.periodic_config.grid_step

        for algorithm, smoothed_periods in self.smoothed_data.items():
            print(f"  Алгоритм: {algorithm}")
            self.interpolated_data[algorithm] = []

            for smoothed_period in smoothed_periods:
                # Используем возрастающие данные для интерполяции
                H_ascending = smoothed_period['H_ascending']
                V_smooth_ascending = smoothed_period['V_smooth_ascending']

                # Создаем равномерную сетку (возрастающую)
                H_min = np.min(H_ascending)
                H_max = np.max(H_ascending)
                # Расширяем сетку на один узел сверху и снизу (в пределах [0, 2R]) -- чтобы не потерять граничные значения
                H_min_ext = max(0.0, H_min - grid_step)
                H_max_ext = min(2.0 * self.geometry.config.R, H_max + grid_step)
                H_grid_ascending = np.arange(H_min_ext, H_max_ext + grid_step, grid_step)

                # Интерполяция на сетку
                # Безопасная линейная интерполяция: используем только валидные (finite) значения
                try:
                    V_arr = np.array(V_smooth_ascending, dtype=float)
                    H_arr = np.array(H_ascending, dtype=float)
                    finite_mask = np.isfinite(V_arr)
                    # Если у нас нет валидных точек — используем V_arr (возможно все NaN) или пропускаем
                    if np.sum(finite_mask) == 0:
                        # ничего валидного — заполним нулями или оставим NaN
                        V_grid_ascending = np.full_like(H_grid_ascending, np.nan, dtype=float)
                    elif np.sum(finite_mask) == 1:
                        # одна точка — заполним константой
                        val = float(V_arr[finite_mask][0])
                        V_grid_ascending = np.full_like(H_grid_ascending, val, dtype=float)
                    else:
                        # Линейная интерполяция внутри диапазона и линейная экстраполяция на краях
                        H_valid = H_arr[finite_mask]
                        V_valid = V_arr[finite_mask]
                        # Убедимся, что H_valid возрастающий
                        sort_idx = np.argsort(H_valid)
                        H_valid = H_valid[sort_idx]
                        V_valid = V_valid[sort_idx]

                        # Если валидных точек одна — заполняем константой
                        if H_valid.size == 1:
                            V_grid_ascending = np.full_like(H_grid_ascending, float(V_valid[0]), dtype=float)
                        else:
                            # Интерполяция внутри диапазона
                            V_interp = np.interp(H_grid_ascending, H_valid, V_valid)
                            # Отключаем экстраполяцию: оставляем значения вне диапазона как NaN
                            V_grid_ascending = V_interp.copy()
                            left_mask = H_grid_ascending < H_valid[0]
                            right_mask = H_grid_ascending > H_valid[-1]
                            if np.any(left_mask):
                                V_grid_ascending[left_mask] = np.nan
                            if np.any(right_mask):
                                V_grid_ascending[right_mask] = np.nan
                except Exception as e:
                    print(f"    Ошибка безопасной интерполяции для периода {smoothed_period.get('period_id', '?')}: {e}")
                    V_grid_ascending = np.full_like(H_grid_ascending, np.nan, dtype=float)

                # Разворачиваем обратно для согласованности с исходными данными
                H_grid = H_grid_ascending[::-1]
                V_grid = V_grid_ascending[::-1]

                # Сохраняем интерполированные данные
                interpolated_period = {
                    'period_id': smoothed_period['period_id'],
                    'H_grid': H_grid,
                    'V_grid': V_grid,
                    'H_original': smoothed_period['H_measured'],
                    'V_smooth_original': smoothed_period['V_smooth']
                }

                self.interpolated_data[algorithm].append(interpolated_period)

    def create_combined_characteristics(self):
        """Создание единой монолитной характеристики V(H) путем усреднения по периодам"""
        print("Создание объединенной характеристики...")

        self.combined_characteristics = {}
        grid_step = self.periodic_config.grid_step

        # Создаем общую сетку высот от 0 до 2R
        H_common = np.arange(0, 2 * self.geometry.config.R + grid_step, grid_step)

        for algorithm in self.periodic_config.smoothing_algorithms:
            print(f"  Алгоритм: {algorithm}")
            self.combined_Pandas[algorithm] = pd.DataFrame()


            # Создаем словарь для хранения объемов по высотам, привязанный к сетке grid_step
            # Ключ — узел сетки (в см), значение — список записей (V, period_id)
            height_volumes = {}

            # Собираем все объемы для каждой высоты из всех периодов
            dropped_total = 0
            per_period_dropped: Dict[int, int] = {}
            for interpolated_period in self.interpolated_data[algorithm]:
                H_grid = interpolated_period['H_grid']
                V_grid = interpolated_period['V_grid']
                period_id = interpolated_period.get('period_id', None)
                per_period_dropped[period_id] = 0

                for h, v in zip(H_grid, V_grid):
                    # Пропускаем нечисловые значения
                    try:
                        v_f = float(v)
                    except Exception:
                        per_period_dropped[period_id] += 1
                        dropped_total += 1
                        continue
                    if not np.isfinite(v_f):
                        per_period_dropped[period_id] += 1
                        dropped_total += 1
                        continue

                    # Привязка узла к сетке через индекс: избегаем ошибок округления FP
                    h_index = int(round(float(h) / grid_step))
                    h_key = float(h_index * grid_step)

                    if h_key not in height_volumes:
                        height_volumes[h_key] = []

                    # Если для этого узла уже есть значение от того же period_id — пропускаем последующие
                    # Храним кортеж (V, period_id) — позже усредним только V
                    if not any((rec[1] == period_id) for rec in height_volumes[h_key]):
                        height_volumes[h_key].append((v_f, period_id))

            # Диагностический дамп: все записи до агрегации (node_cm, V, period_id)
            try:
                dump_dir = os.path.join('plots', self.plot_subdir)
                os.makedirs(dump_dir, exist_ok=True)
                dump_path = os.path.join(dump_dir, f'cloud_before_agg_{algorithm}.csv')
                with open(dump_path, 'w', newline='', encoding='utf-8') as f:
                    import csv
                    w = csv.writer(f)
                    w.writerow(['node_cm', 'V', 'period_id'])
                    for node in sorted(height_volumes.keys()):
                        for rec in height_volumes[node]:
                            v_val = rec[0] if isinstance(rec, (list, tuple)) else rec
                            pid = rec[1] if isinstance(rec, (list, tuple)) and len(rec) > 1 else ''
                            w.writerow([node, float(v_val), pid])
                print(f"    Диагностический дамп облака сохранён: {dump_path}")
            except Exception as e:
                print(f"    Ошибка при сохранении дампа облака: {e}")

            # Создаем массивы для объединенной характеристики: берем все узлы, где есть хотя бы одна запись
            H_combined = []
            V_combined = []

            # Проходим по отсортированным узлам (в пределах общей сетки H_common)
            for h_key in sorted(height_volumes.keys()):
                # Опционально фильтруем узлы вне общей сетки (на случай интерполяционных краёв)
                if h_key < np.min(H_common) - 1e-8 or h_key > np.max(H_common) + 1e-8:
                    continue

                recs = height_volumes[h_key]
                # берем только первые элементы кортежей — объемы
                vols = [rec[0] if isinstance(rec, (list, tuple)) else rec for rec in recs]
                if len(vols) == 0:
                    continue
                avg_volume = float(np.mean(vols))
                H_combined.append(h_key)
                V_combined.append(avg_volume)

            # Преобразуем в numpy массивы
            H_combined = np.array(H_combined)
            V_combined = np.array(V_combined)

            # Сохраняем необработанную (raw) версию для диагностики
            V_raw = V_combined.copy()

            # DEBUG_BREAKPOINT: здесь можно поставить точку останова, чтобы проинспектировать
            # переменную `height_volumes` или файл дампа cloud_before_agg_{algorithm}.csv
            # для визуальной проверки облака значений перед агрегацией.

            # Простое сглаживание объединённой характеристики тем же алгоритмом SciPy (UnivariateSpline)
            # используем ту же функцию `scipy_smooth`, что применяется к периодам.
            V_smoothed = V_combined.copy()
            if H_combined.size >= 2:
                try:
                    # scipy_smooth ожидает индексы I_beg/I_end для диапазона сглаживания
                    V_smoothed = scipy_smooth(H_combined, V_smoothed, 0, len(H_combined) - 1, smooth_factor=0.1)
                except Exception as e:
                    print(f"    Ошибка при сглаживании объединённой характеристики ({algorithm}): {e}")
                    V_smoothed = V_raw.copy()

            # Дополнительно: усиленное сглаживание (чтобы сравнить эффект)
            V_smoothed_stronger = V_raw.copy()
            stronger_factor = getattr(self.periodic_config, 'combined_smooth_factor', None)
            if stronger_factor is None:
                # значение по умолчанию для сильного сглаживания
                stronger_factor = 1.0

            if H_combined.size >= 2:
                try:
                    V_smoothed_stronger = scipy_smooth(H_combined, V_smoothed_stronger, 0, len(H_combined) - 1,
                                                        smooth_factor=float(stronger_factor))
                except Exception as e:
                    print(f"    Ошибка при усиленном сглаживании объединённой характеристики ({algorithm}): {e}")
                    V_smoothed_stronger = V_raw.copy()

            # Сохраняем объединенную (сглаженную) характеристику и raw-версию
            self.combined_characteristics[algorithm] = {
                'H': H_combined,
                'V': V_smoothed,
                'V_raw': V_raw,
                'V_stronger': V_smoothed_stronger,
                'periods_count': len(height_volumes)
            }

            # Дополнительно сохраняем объединённую характеристику и counts в CSV для удобного анализа
            try:
                combined_path = os.path.join('plots', self.plot_subdir, f'combined_{algorithm}.csv')
                # В файл combined записываем именно сглаженную версию объединённой характеристики
                try:
                    with open(combined_path, 'w', newline='', encoding='utf-8') as f2:
                        import csv
                        ww = csv.writer(f2)
                        ww.writerow(['H_cm', 'V_l'])
                        # берем значения из сохранённой сглаженной версии
                        V_to_write = self.combined_characteristics[algorithm].get('V', V_raw)
                        for hh, vv in zip(H_combined, V_to_write):
                            ww.writerow([hh, vv])
                except Exception as e:
                    print(f"    Ошибка при сохранении combined CSV (smoothing): {e}")
                counts_path = os.path.join('plots', self.plot_subdir, f'counts_{algorithm}.csv')
                try:
                    with open(counts_path, 'w', newline='', encoding='utf-8') as f3:
                        import csv
                        ww = csv.writer(f3)
                        ww.writerow(['node_cm', 'count'])
                        for node in sorted(height_volumes.keys()):
                            ww.writerow([node, len(height_volumes[node])])
                    print(f"    Сохранены combined CSV: {combined_path} и counts CSV: {counts_path}")
                except Exception as e:
                    print(f"    Не удалось сохранить counts CSV: {e}")
            except Exception as e:
                print(f"    Не удалось сохранить combined/counts CSV: {e}")
            # Сохраняем также файл с усиленным сглаживанием и график сравнения
            try:
                print(f"    Попытка сохранить усиленное сглаживание для алгоритма {algorithm} в {os.path.join('plots', self.plot_subdir)}")
                os.makedirs(os.path.join('plots', self.plot_subdir), exist_ok=True)
                combined_strong_path = os.path.join('plots', self.plot_subdir, f'combined_{algorithm}_strong.csv')
                with open(combined_strong_path, 'w', newline='', encoding='utf-8') as f4:
                    import csv
                    ww = csv.writer(f4)
                    ww.writerow(['H_cm', 'V_raw_l', 'V_smoothed_l', 'V_smoothed_stronger_l'])
                    for hh, vraw, vs, vst in zip(H_combined, V_raw, V_smoothed, V_smoothed_stronger):
                        ww.writerow([hh, vraw, vs, vst])

                # Построим сравнительный график raw / original smooth / stronger smooth
                try:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(H_combined, V_raw, color='gray', linewidth=0.8, alpha=0.8, label='Raw average')
                    ax.plot(H_combined, V_smoothed, color='blue', linewidth=1.2, alpha=0.9, label='Smoothed (orig)')
                    ax.plot(H_combined, V_smoothed_stronger, color='red', linewidth=2.0, alpha=0.9, label='Smoothed (strong)')
                    ax.set_xlabel('Уровень жидкости, см')
                    ax.set_ylabel('Объем, л')
                    ax.set_title(f'Combined smoothing comparison - {algorithm}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    comp_path = os.path.join('plots', self.plot_subdir, f'combined_comparison_{algorithm}.png')
                    plt.tight_layout()
                    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    Сохранён сравнительный график сглаживаний: {comp_path}")
                except Exception as e:
                    print(f"    Не удалось построить сравнительный график: {e}")
            except Exception as e:
                print(f"    Не удалось сохранить combined strong CSV: {e}")

        # Диагностический вывод глобального min/max высот
        for algorithm in self.periodic_config.smoothing_algorithms:
            if algorithm in self.combined_characteristics:
                combined_data = self.combined_characteristics[algorithm]
                H_min_global = np.min(combined_data['H'])
                H_max_global = np.max(combined_data['H'])
                print(f"  Алгоритм {algorithm}: глобальный min/max высот = {H_min_global:.2f} см / {H_max_global:.2f} см")

    def visualize_combined_characteristics(self):
        """Сравнение разных алгоритмов сглаживания для объединенной характеристики"""
        if len(self.combined_characteristics) == 0:
            return

        plt.figure(figsize=(12, 8))

        # Цвета для алгоритмов
        algorithm_colors = {
            'scipy': 'red',
            'original_sg1': 'blue',
            'original_sg2': 'green'
        }

        # Рисуем объединенные характеристики для каждого алгоритма
        for algorithm in self.periodic_config.smoothing_algorithms:
            if algorithm in self.combined_characteristics:
                combined_data = self.combined_characteristics[algorithm]

                algorithm_names = {
                    'scipy': 'SciPy',
                    'original_sg1': 'Оригинальный Sg_p=1',
                    'original_sg2': 'Оригинальный Sg_p=2'
                }

                # безопасно получаем количество точек
                n_points = len(combined_data['H']) if (combined_data.get('H') is not None) else 0
                markevery_val = max(1, n_points // 20) if n_points > 0 else 1
                plt.plot(combined_data['H'], combined_data['V'],
                         color=algorithm_colors.get(algorithm, 'black'),
                         linewidth=2,
                         label=f'{algorithm_names.get(algorithm, algorithm)} ({combined_data["periods_count"]} периодов)',
                         linestyle='--',
                         marker='o',
                         markersize=4,
                         markevery=markevery_val)

        plt.xlabel('Уровень жидкости, см', fontsize=12)
        plt.ylabel('Объем, л', fontsize=12)
        plt.title(f'Сравнение объединенных характеристик\n{self.name}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Сохранение графика
        filename = f'algorithms_comparison_period1.png'
        filepath = f'plots/{self.plot_subdir}/{filename}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   График сравнения алгоритмов сохранен: {filepath}")

    def visualize_combined_vs_geometry(self):
        """Сравнение объединённой характеристики и точной геометрической градуировки для каждого алгоритма.
        Сохраняет по одному PNG на алгоритм в папке эксперимента.
        """
        if len(self.combined_characteristics) == 0:
            return

        # Получаем идеальную геометрическую кривую
        H_geom_full, V_geom_full = self.geometry.calculate_ideal_curve()

        algorithm_names = {
            'scipy': 'SciPy',
            'original_sg1': 'Оригинальный Sg_p=1',
            'original_sg2': 'Оригинальный Sg_p=2'
        }

        for algorithm, combined in self.combined_characteristics.items():
            Hc = combined.get('H', np.array([]))
            Vc = combined.get('V', np.array([]))

            # Если есть объединённая характеристика, обрезаем идеальную кривую по её min/max
            if Hc.size > 0:
                h_min = float(np.min(Hc))
                h_max = float(np.max(Hc))
                mask = (H_geom_full >= h_min - 1e-8) & (H_geom_full <= h_max + 1e-8)
                if np.any(mask):
                    H_geom = H_geom_full[mask]
                    V_geom = V_geom_full[mask]
                else:
                    H_geom = H_geom_full
                    V_geom = V_geom_full
            else:
                H_geom = H_geom_full
                V_geom = V_geom_full

            fig, ax = plt.subplots(figsize=(10, 6))

            # ПИГЛ — зелёная линия (без маркеров)
            ax.plot(H_geom, V_geom, color='green', linewidth=1.5, alpha=0.9, label='ПИГЛ')

            # Алгоритм — красная пунктирная линия (без маркеров), сохраняем пунктирность и прозрачность
            if Hc.size > 0:
                ax.plot(Hc, Vc, color='red', linestyle='--', linewidth=1.2, alpha=0.9,
                        label=algorithm_names.get(algorithm, algorithm))

            # Заголовки и подписи
            fig.suptitle('Зависимость объёма от уровня', fontsize=14)
            ax.set_title(f'fСравнение ПИГЛ - {algorithm_names.get(algorithm, algorithm)}', fontsize=12)
            ax.set_xlabel('H, см', fontsize=12)
            ax.set_ylabel('V, л', fontsize=12)

            # Легенда
            ax.legend(fontsize=10)

            # Основная сетка (более заметная)
            ax.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.6)
            # Включаем минорные деления и рисуем пунктирную сантиметровую сетку (без подписей)
            try:
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth=0.6, alpha=0.3)
            except Exception:
                # Если по какой-то причине minor ticks не поддержаны — игнорируем
                pass

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            filename = f'combined_vs_geometry_{algorithm}.png'
            filepath = f'plots/{self.plot_subdir}/{filename}'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"   Сохранён график сравнения combined vs geometry: {filepath}")

    def visualize_periods(self):
        """Визуализация уровней от глобального шага"""
        plt.figure(figsize=(16, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.periods_data)))

        # Собираем все точки для сквозного графика
        all_steps = []
        all_levels = []
        period_boundaries = []  # Границы периодов для вертикальных линий

        for i, period in enumerate(self.periods_data):
            # Собираем данные для текущего периода
            steps = [point['global_step'] for point in period['points']]
            levels = [point['H_measured'] for point in period['points']]

            # Добавляем в общие массивы
            all_steps.extend(steps)
            all_levels.extend(levels)

            # Отмечаем границы периодов
            if steps:  # Если период не пустой
                period_boundaries.append(steps[0])  # Начало периода

            # Рисуем линию периода
            plt.plot(steps, levels,
                     color=colors[i],
                     marker='o',
                     markersize=4,
                     linewidth=2,
                     label=f'Период {period["period_id"] + 1}',
                     alpha=0.8)

        # Добавляем вертикальные линии для разделения периодов
        for boundary in period_boundaries[1:]:  # Пропускаем первую границу (начало эксперимента)
            plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.5, linewidth=1)

        plt.xlabel('Глобальный номер шага', fontsize=12)
        plt.ylabel('Уровень жидкости, см', fontsize=12)
        plt.title(f'Динамика уровней жидкости по периодам опустошения\n{self.name}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Настройка внешнего вида
        plt.tight_layout()

        # Сохранение графика
        filename = f'periodic_draining_levels.png'
        filepath = f'plots/{self.plot_subdir}/{filename}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   График сохранен: {filepath}")

    def visualize_smoothed_curves(self):
        """Визуализация сглаженных кривых V(H) для всех алгоритмов"""
        # Создаем подграфики для каждого алгоритма
        n_algorithms = len(self.periodic_config.smoothing_algorithms)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 8))

        if n_algorithms == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.periods_data)))

        for idx, algorithm in enumerate(self.periodic_config.smoothing_algorithms):
            ax = axes[idx]

            # Рисуем сглаженные кривые для каждого периода (полупрозрачные)
            for i, interpolated_period in enumerate(self.interpolated_data[algorithm]):
                H_grid = interpolated_period['H_grid']
                V_grid = interpolated_period['V_grid']

                # Делаем линии периодов непрозрачными и добавляем их в легенду
                ax.plot(H_grid, V_grid,
                        color=colors[i],
                        linewidth=1,
                        alpha=1.0,
                        label=f'Период {i + 1}')

            # Рисуем объединенную характеристику (жирная линия)
            if algorithm in self.combined_characteristics:
                combined_data = self.combined_characteristics[algorithm]
                # Рассчитываем параметр markevery в зависимости от числа точек, чтобы маркеров было не слишком много
                # безопасно получаем количество точек
                n_points = len(combined_data['H']) if (combined_data.get('H') is not None) else 0
                markevery_val = max(1, n_points // 20) if n_points > 0 else 1
                ax.plot(combined_data['H'], combined_data['V'],
                        color='black',
                        linewidth=1,
                        alpha=1.0,
                        label='Объединенная характеристика',
                        linestyle='--',
                        marker='o',
                        markersize=4,
                        markerfacecolor='black',
                        markeredgecolor='black',
                        markevery=markevery_val)

            # Настройки графика
            algorithm_names = {
                'scipy': 'SciPy',
                'original_sg1': 'Оригинальный Sg_p=1',
                'original_sg2': 'Оригинальный Sg_p=2'
            }

            ax.set_xlabel('Уровень жидкости, см', fontsize=12)
            ax.set_ylabel('Объем, л', fontsize=12)
            ax.set_title(f'{algorithm_names.get(algorithm, algorithm)}\n{self.name}', fontsize=14)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Сохранение графика
        filename = f'smoothed_curves_comparison.png'
        filepath = f'plots/{self.plot_subdir}/{filename}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   График сглаженных кривых сохранен: {filepath}")

        # Дополнительный график: сравнение всех алгоритмов для объединенной характеристики
        self.visualize_combined_characteristics()

    def visualize_periods_before_extrapolation(self):
        """Сохраняет график сглаженных кривых каждого периода на их исходных узлах (до интерполяции/экстраполяции).
        В файле не рисуются объединённые характеристики и не используется H_grid/V_grid.
        """
        if not self.smoothed_data:
            return

        n_algorithms = len(self.periodic_config.smoothing_algorithms)
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 6))
        if n_algorithms == 1:
            axes = [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(self.periods_data))))

        for idx, algorithm in enumerate(self.periodic_config.smoothing_algorithms):
            ax = axes[idx]
            ax.set_title(f'{algorithm} — {self.name}', fontsize=12)

            # Рисуем сглаженные кривые каждого периода на их исходных уникальных узлах
            for i, smoothed_period in enumerate(self.smoothed_data.get(algorithm, [])):
                H_asc = smoothed_period.get('H_ascending', np.array([]))
                V_smooth_asc = smoothed_period.get('V_smooth_ascending', np.array([]))
                if H_asc is None or V_smooth_asc is None:
                    continue
                H_arr = np.array(H_asc, dtype=float)
                V_arr = np.array(V_smooth_asc, dtype=float)
                if H_arr.size == 0 or V_arr.size == 0:
                    continue
                # Рисуем непрозрачные линии периодов и добавляем подпись
                ax.plot(H_arr, V_arr, color=colors[i % len(colors)], linewidth=1.5, alpha=1.0, label=f'Период {i+1}')

            ax.set_xlabel('Уровень жидкости, см', fontsize=11)
            ax.set_ylabel('Объем, л', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.tight_layout()
        filename = 'smoothed_curves_periods_only.png'
        filepath = os.path.join('plots', self.plot_subdir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   График периодов до интерполяции сохранён: {filepath}")

    def visualize_derivative_for_combined(self, algorithm: str):
        """Построение производной объединённой характеристики и сравнение с точной геометрией.

        Здесь производная считается непосредственно по уже усреднённой кривой (Hc, Vc).
        Для сравнения идеальную кривую интерполируем на те же H и считаем её производную.
        """
        if algorithm not in self.combined_characteristics:
            return

        combined = self.combined_characteristics[algorithm]
        Hc = combined.get('H', np.array([]))
        Vc = combined.get('V', np.array([]))
        if Hc.size == 0 or Vc.size == 0:
            return

        # Сортируем H/V по возрастанию H
        sort_idx = np.argsort(Hc)
        Hc_s = Hc[sort_idx]
        Vc_s = Vc[sort_idx]

        if Hc_s.size < 2:
            return

        # Идеальная кривая (полная)
        H_geom_full, V_geom_full = self.geometry.calculate_ideal_curve()

        # Интерполируем идеальную кривую на узлы объединённой характеристики
        V_geom_on_Hc = np.interp(Hc_s, H_geom_full, V_geom_full)

        # Вычисляем производные напрямую на узлах объединённой характеристики
        dV_comb = calculate_derivative(Hc_s, Vc_s)
        dV_ideal_on_Hc = calculate_derivative(Hc_s, V_geom_on_Hc)

        # Ошибка производной (Del_V2) в процентах
        eps = 1e-10
        Del_V2 = ((dV_ideal_on_Hc - dV_comb) / (dV_ideal_on_Hc + eps)) * 100

        # Рисуем график ошибок и значений производных
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax1.plot(Hc_s, Del_V2, 'r--', label='Del_V2 - ошибка производной объединённой характеристики')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Ошибка производной (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Ошибка производной объединённой характеристики')

        ax2.plot(Hc_s, dV_ideal_on_Hc, 'b-', label='dV - идеальная производная (на H_combined)')
        ax2.plot(Hc_s, dV_comb, 'r--', label=f'dV - {algorithm} (объединённая)')
        ax2.set_xlabel('H, см')
        ax2.set_ylabel('Производная объёма (л/см)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'derivative_analysis_{algorithm}_combined.png'
        filepath = os.path.join('plots', self.plot_subdir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f"   Сохранён график производной (combined): {filepath}")

    def visualize_derivatives_all(self):
        """Визуализация производных для всех алгоритмов: и объединённой характеристики, и периодов."""
        for algorithm in self.periodic_config.smoothing_algorithms:
            try:
                self.visualize_derivative_for_combined(algorithm)
            except Exception as e:
                print(f"Ошибка при построении производной для combined {algorithm}: {e}")
            # Примечание: производные по отдельным периодам не строим — используем только усреднённую кривую

    def print_periods_statistics(self):
        """Вывод статистики по периодам"""
        print("\n" + "=" * 50)
        print("СТАТИСТИКА ПЕРИОДОВ")
        print("=" * 50)

        total_points = 0
        total_global_steps = 0

        for period in self.periods_data:
            n_points = len(period['points'])
            total_points += n_points
            start_level = period['start_level']
            end_level = period['end_level']
            final_level = period['points'][-1]['H_measured']
            final_level_ideal = period['points'][-1]['H_ideal_current']
            total_drain = start_level - end_level

            # Статистика по шагам опустошения (исключая начальную точку)
            drain_steps = [point['drain_step'] for point in period['points'][1:]]
            avg_drain_step = np.mean(drain_steps) if drain_steps else 0

            # Глобальные шаги
            global_steps = [point['global_step'] for point in period['points']]
            period_global_range = f"{min(global_steps)}-{max(global_steps)}"

            print(f"Период {period['period_id'] + 1}:")
            print(f"  Начальный уровень: {start_level:.1f} см")
            print(f"  Конечный уровень: {end_level:.1f} см (измеренный: {final_level:.1f} см)")
            print(f"  Количество шагов: {n_points}")
            print(f"  Глобальные шаги: {period_global_range}")
            print(f"  Общее опустошение: {total_drain:.1f} см")
            print(f"  Средний шаг опустошения: {avg_drain_step:.1f} см")

            # Информация о первом и последнем измерениях
            first_point = period['points'][0]
            last_point = period['points'][-1]
            print(
                f"  Начальный объем: {first_point['V_measured']:.1f} л (точный: {first_point['V_ideal_current']:.1f} л)")
            print(f"  Конечный объем: {last_point['V_measured']:.1f} л (точный: {last_point['V_ideal_current']:.1f} л)")
            print()

            total_global_steps = max(global_steps)

        print(f"ОБЩАЯ СТАТИСТИКА:")
        print(f"  Всего периодов: {len(self.periods_data)}")
        print(f"  Всего точек измерений: {total_points}")
        print(f"  Всего глобальных шагов: {total_global_steps + 1}")
        print(f"  Средняя длина периода: {total_points / len(self.periods_data):.1f} шагов")

    def get_experiment_data(self) -> Tuple[Any, Any, Any, Any, Any, List[Dict]]:
        """
        Получение данных эксперимента.
        Возвращает данные периодов.
        """
        return (None, None, None, None, None, self.periods_data)

    def get_detailed_period_data(self, period_id: int) -> List[Dict]:
        """Получение детальных данных по конкретному периоду"""
        if 0 <= period_id < len(self.periods_data):
            return self.periods_data[period_id]['points']
        return []
