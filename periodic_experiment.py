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
from utils import OriginalSmoother, scipy_smooth, SafeOriginalSmoother


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

            # 6. Интерполяция на сетку
            print(f"\nИНТЕРПОЛЯЦИЯ НА СЕТКУ...")
            self.interpolate_to_grid()

            # 7. Создание объединенной характеристики
            print(f"\nСОЗДАНИЕ ОБЪЕДИНЕННОЙ ХАРАКТЕРИСТИКИ...")
            self.create_combined_characteristics()

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

                # Применяем выбранный алгоритм сглаживания
                V_smooth_ascending = self.apply_smoothing_algorithm(algorithm, H_ascending, V_ascending)

                # Разворачиваем обратно
                V_smooth = V_smooth_ascending[::-1]

                # Сохраняем сглаженные данные
                smoothed_period = {
                    'period_id': period['period_id'],
                    'H_measured': H_measured,
                    'V_measured': V_measured,
                    'V_smooth': V_smooth,
                    'H_ascending': H_ascending,
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
                H_grid_ascending = np.arange(H_min, H_max + grid_step, grid_step)

                # Интерполяция на сетку
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Убедимся, что данные строго возрастают
                        if len(np.unique(H_ascending)) == len(H_ascending):
                            spline = CubicSpline(H_ascending, V_smooth_ascending)
                            V_grid_ascending = spline(H_grid_ascending)
                        else:
                            # Если есть дубликаты, используем линейную интерполяцию
                            V_grid_ascending = np.interp(H_grid_ascending, H_ascending, V_smooth_ascending)
                except Exception as e:
                    print(f"    Ошибка интерполяции для периода {smoothed_period['period_id']}: {e}")
                    # Линейная интерполяция как запасной вариант
                    V_grid_ascending = np.interp(H_grid_ascending, H_ascending, V_smooth_ascending)

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


            # Создаем словарь для хранения объемов по высотам
            height_volumes = {}

            # Собираем все объемы для каждой высоты из всех периодов
            for interpolated_period in self.interpolated_data[algorithm]:
                H_grid = interpolated_period['H_grid']
                V_grid = interpolated_period['V_grid']

                for h, v in zip(H_grid, V_grid):
                    # Округляем высоту до сетки
                    h_rounded = self.round_level(h)

                    if h_rounded not in height_volumes:
                        height_volumes[h_rounded] = []

                    height_volumes[h_rounded].append(v)

            # Создаем массивы для объединенной характеристики
            H_combined = []
            V_combined = []

            # Проходим по общей сетке высот
            for h in H_common:
                h_rounded = self.round_level(h)

                if h_rounded in height_volumes:
                    volumes = height_volumes[h_rounded]
                    # Усредняем объемы по всем периодам
                    avg_volume = np.mean(volumes)
                    H_combined.append(h_rounded)
                    V_combined.append(avg_volume)

            # Преобразуем в numpy массивы
            H_combined = np.array(H_combined)
            V_combined = np.array(V_combined)

            # Сохраняем объединенную характеристику
            self.combined_characteristics[algorithm] = {
                'H': H_combined,
                'V': V_combined,
                'periods_count': len(height_volumes)
            }
            # self.combined_Pandas[algorithm][interpolated_period]

            print(f"    Создана характеристика с {len(H_combined)} точками")

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
        plt.show()

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

                ax.plot(H_grid, V_grid,
                        color=colors[i],
                        linewidth=1,
                        alpha=0.3)  # Полупрозрачные линии периодов

            # Рисуем объединенную характеристику (жирная линия)
            if algorithm in self.combined_characteristics:
                combined_data = self.combined_characteristics[algorithm]
                ax.plot(combined_data['H'], combined_data['V'],
                        color='black',
                        linewidth=3,
                        label='Объединенная характеристика')

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
        plt.show()

        print(f"   График сглаженных кривых сохранен: {filepath}")

        # Дополнительный график: сравнение всех алгоритмов для объединенной характеристики
        self.visualize_combined_characteristics()

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

                plt.plot(combined_data['H'], combined_data['V'],
                         color=algorithm_colors.get(algorithm, 'black'),
                         linewidth=2,
                         label=f'{algorithm_names.get(algorithm, algorithm)} ({combined_data["periods_count"]} периодов)')

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
        plt.show()

        print(f"   График сравнения алгоритмов сохранен: {filepath}")

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