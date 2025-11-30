# periodic_experiment.py
"""
Эксперимент с периодами опустошения цилиндра.
Наследуется от основного класса CylinderExperiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple, Any

from experiment import CylinderExperiment
from config import PeriodicExperimentConfig
from geometry import CylinderGeometry


class PeriodicDrainingExperiment(CylinderExperiment):
    """
    Эксперимент с периодами опустошения цилиндра.
    Наследует основную функциональность от CylinderExperiment.
    """

    def __init__(self, config: PeriodicExperimentConfig, name: str = ""):
        super().__init__(config, name)
        self.periodic_config = config
        self.periods_data = []  # Список периодов
        self.rng = np.random.RandomState(config.seed)
        self.global_step_counter = 0  # Сквозной счётчик шагов

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

        try:
            # 1. Инициализация геометрии (как в родительском классе)
            self.config.print_config()
            self.geometry.print_geometry_info()

            # 2. Генерация периодов опустошения
            print(f"\nГЕНЕРАЦИЯ {self.periodic_config.n_periods} ПЕРИОДОВ ОПУСТОШЕНИЯ...")
            self.periods_data = self.generate_periods()

            # 3. Вывод статистики
            self.print_periods_statistics()

            # 4. Визуализация (если включено)
            if self.periodic_config.enable_period_plotting:
                print("\nСОЗДАНИЕ ГРАФИКОВ...")
                self.visualize_periods()

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
        os.makedirs('plots', exist_ok=True)
        filename = f'periodic_draining_{self.name.replace(" ", "_").replace("-", "_")}.png'
        filepath = f'plots/{filename}'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"   График сохранен: {filepath}")

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