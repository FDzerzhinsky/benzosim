# main.py
"""
Главный скрипт для запуска непериодических экспериментов с цилиндром.
Для периодических экспериментов используйте main_periodic.py
"""

from experiment import CylinderExperiment
from visualization import ExperimentVisualizer, DataExporter
import os
from config import ExperimentConfig, ObstacleType, NON_PERIODIC_CONFIGS
from typing import Dict


def main():
    """
    Основная функция запуска экспериментов.
    Запускает непериодические эксперименты.
    """
    print("ЭКСПЕРИМЕНТ С ЦИЛИНДРОМ НА БОКУ (НЕПЕРИОДИЧЕСКИЙ)")
    print("=" * 60)

    experiments_data = {}
    experiment_objects = {}

    try:
        # Запуск стандартных (непериодических) экспериментов
        run_standard_experiments(experiments_data, experiment_objects)

    except Exception as e:
        print(f"Ошибка в основном цикле: {e}")
        import traceback
        traceback.print_exc()

    # ФИНАЛЬНЫЙ ОТЧЕТ
    _print_final_report(experiments_data)


def run_standard_experiments(experiments_data: Dict, experiment_objects: Dict):
    """Запуск непериодических экспериментов из NON_PERIODIC_CONFIGS"""
    print("\nЗАПУСК НЕПЕРИОДИЧЕСКИХ ЭКСПЕРИМЕНТОВ")
    print("=" * 50)

    # Используем конфигурации, определённые в config.py
    for config in NON_PERIODIC_CONFIGS:
        # Формируем путь для сохранения графиков
        output_dir = os.path.join('plots', 'non_periodic', config.name)
        os.makedirs(output_dir, exist_ok=True)
        # Сохраняем путь в конфигурацию (добавляем атрибут динамически)
        setattr(config, 'output_dir', output_dir)

        experiment_name = f"NonPeriodic - {config.name}"
        print(f"\n--- {experiment_name} ---")

        experiment = CylinderExperiment(config, experiment_name)
        data = experiment.run_experiment()

        if data:
            experiments_data[experiment_name] = data
            experiment_objects[experiment_name] = experiment


def run_periodic_experiments(experiments_data: Dict, experiment_objects: Dict):
    """
    Запуск периодических экспериментов.
    Примечание: Для периодических экспериментов используйте main_periodic.py
    """
    print("\nВНИМАНИЕ: Для периодических экспериментов используйте main_periodic.py")
    print("Этот файл (main.py) предназначен для непериодических экспериментов.")


def _print_final_report(experiments_data: Dict):
    """Печать финального отчета"""
    print("\n" + "=" * 60)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("=" * 60)

    print(f"Успешно выполнено экспериментов: {len(experiments_data)}")

    if experiments_data:
        print("\nСозданные графики в папке 'plots/':")
        for exp_name in experiments_data.keys():
            safe_name = exp_name.replace(" ", "_").replace("-", "_")
            print(f"  {safe_name}.png")


if __name__ == "__main__":
    # Запуск полного набора экспериментов
    main()
