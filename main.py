# main.py
"""
Главный скрипт для запуска экспериментов с цилиндром.
Теперь с поддержкой периодических экспериментов.
"""

from experiment import CylinderExperiment
from visualization import ExperimentVisualizer, DataExporter
from config import ExperimentConfig, ObstacleType
from periodic_experiment import PeriodicDrainingExperiment, PeriodicExperimentConfig
import numpy as np
from typing import Dict, Tuple


def main():
    """
    Основная функция запуска экспериментов.
    Запускает все комбинации алгоритмов и типов помех.
    """
    print("ЭКСПЕРИМЕНТ С ЦИЛИНДРОМ НА БОКУ")
    print("=" * 60)

    experiments_data = {}
    experiment_objects = {}

    try:
        # ВАРИАНТ 1: Обычные эксперименты (раскомментировать при необходимости)
        # run_standard_experiments(experiments_data, experiment_objects)

        # ВАРИАНТ 2: Периодические эксперименты
        run_periodic_experiments(experiments_data, experiment_objects)

    except Exception as e:
        print(f"Ошибка в основном цикле: {e}")
        import traceback
        traceback.print_exc()

    # ФИНАЛЬНЫЙ ОТЧЕТ
    _print_final_report(experiments_data)


def run_standard_experiments(experiments_data: Dict, experiment_objects: Dict):
    """Запуск стандартных экспериментов"""
    # [существующий код запуска стандартных экспериментов]
    pass


def run_periodic_experiments(experiments_data: Dict, experiment_objects: Dict):
    """Запуск периодических экспериментов"""
    print("\nЗАПУСК ПЕРИОДИЧЕСКИХ ЭКСПЕРИМЕНТОВ")
    print("=" * 50)

    # Конфигурации для разных типов помех
    periodic_configs = [
        PeriodicExperimentConfig(
            obstacle_type=ObstacleType.NONE,
            n_periods=3,
            refill_min_level=60.0,
            refill_max_level=170.0,
            min_final_level=15.0,
            min_drain_step=2.0,
            max_drain_step=8.0,
            level_precision=1,
            enable_period_plotting=True,
            name="Без помех"
        ),
        PeriodicExperimentConfig(
            obstacle_type=ObstacleType.PARALLELEPIPED,
            n_periods=4,
            refill_min_level=80.0,
            refill_max_level=160.0,
            min_final_level=20.0,
            min_drain_step=1.5,
            max_drain_step=6.0,
            level_precision=1,
            enable_period_plotting=True,
            name="С параллелепипедом"
        ),
        PeriodicExperimentConfig(
            obstacle_type=ObstacleType.CYLINDER,
            n_periods=3,
            refill_min_level=70.0,
            refill_max_level=150.0,
            min_final_level=25.0,
            min_drain_step=2.0,
            max_drain_step=7.0,
            level_precision=2,
            enable_period_plotting=True,
            name="С цилиндрической помехой"
        ),
        PeriodicExperimentConfig(
            obstacle_type=ObstacleType.MULTIPLE_PARALLELEPIPEDS,
            n_periods=5,
            refill_min_level=50.0,
            refill_max_level=180.0,
            min_final_level=10.0,
            min_drain_step=3.0,
            max_drain_step=10.0,
            level_precision=1,
            enable_period_plotting=True,
            name="С 4 параллелепипедами"
        )
    ]

    for config in periodic_configs:
        experiment_name = f"Периодический - {config.name}"
        print(f"\n--- {experiment_name} ---")

        experiment = PeriodicDrainingExperiment(config, experiment_name)
        data = experiment.run_experiment()

        if data:
            experiments_data[experiment_name] = data
            experiment_objects[experiment_name] = experiment


def _print_final_report(experiments_data: Dict):
    """Печать финального отчета"""
    print("\n" + "=" * 60)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("=" * 60)

    print(f"Успешно выполнено экспериментов: {len(experiments_data)}")

    # Фильтруем периодические эксперименты
    periodic_experiments = {k: v for k, v in experiments_data.items() if "Периодический" in k}

    if periodic_experiments:
        print(f"Периодических экспериментов: {len(periodic_experiments)}")
        print("\nСозданные графики в папке 'plots/':")
        for exp_name in periodic_experiments.keys():
            safe_name = exp_name.replace(" ", "_").replace("-", "_")
            print(f"  periodic_draining_{safe_name}.png")


if __name__ == "__main__":
    # Запуск полного набора экспериментов
    main()