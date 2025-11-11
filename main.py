# main.py
"""
Главный скрипт для запуска экспериментов с цилиндром.
Соответствует оригинальному Pascal-коду по выводу данных и графиков.
"""

from experiment import CylinderExperiment
from visualization import ExperimentVisualizer, DataExporter
from config import ExperimentConfig, ObstacleType
import numpy as np
from typing import Dict, Tuple
from bspline_new import bspline_


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
        # ВСЕ КОМБИНАЦИИ: 3 алгоритма × 4 типа помех
        obstacle_types = [
            ObstacleType.NONE,
            ObstacleType.PARALLELEPIPED,
            ObstacleType.CYLINDER,
            ObstacleType.MULTIPLE_PARALLELEPIPEDS  # НОВЫЙ ТИП
        ]

        # ТРИ АЛГОРИТМА: SciPy + 2 оригинальных режима
        algorithms = [
            (False, 1, "SciPy"),  # SciPy алгоритм (Sg_p не используется)
            (True, 1, "оригинальный Sg_p=1"),  # Оригинальный метод конечных разностей
            (True, 2, "оригинальный Sg_p=2")   # Оригинальный метод сплайнов
        ]

        for use_original, sg_p, algorithm_name in algorithms:
            for obstacle_type in obstacle_types:
                # Формируем название эксперимента
                obstacle_name = {
                    ObstacleType.NONE: "без помех",
                    ObstacleType.PARALLELEPIPED: "с параллелепипедом",
                    ObstacleType.CYLINDER: "с цилиндрической помехой",
                    ObstacleType.MULTIPLE_PARALLELEPIPEDS: "с 4 параллелепипедами"  # НОВОЕ НАЗВАНИЕ
                }[obstacle_type]

                experiment_name = f"{obstacle_name} - {algorithm_name}"

                print("\n" + "=" * 60)
                print(f"ЭКСПЕРИМЕНТ: {experiment_name}")
                print("=" * 60)

                # Создаем конфигурацию с указанием алгоритма и режима
                config = ExperimentConfig(
                    obstacle_type=obstacle_type,
                    use_original_smoothing=use_original,
                    Sg_p=sg_p  # Указываем режим сглаживания для оригинального алгоритма
                )

                # Запускаем эксперимент
                experiment = CylinderExperiment(config, experiment_name)
                data = experiment.run_experiment()

                if data:
                    experiments_data[experiment_name] = data
                    experiment_objects[experiment_name] = experiment

        # СОЗДАНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ
        if experiments_data and experiment_objects:
            print("\n" + "=" * 60)
            print("СОЗДАНИЕ СРАВНИТЕЛЬНЫХ ГРАФИКОВ")
            print("=" * 60)

            # Используем первый эксперимент для инициализации визуализатора
            first_experiment_name = next(iter(experiment_objects.keys()))
            first_experiment = experiment_objects[first_experiment_name]
            visualizer = ExperimentVisualizer(first_experiment.results)

            # Сравнительные графики
            visualizer.create_comparison_plots(experiments_data, experiment_objects)

        # ЭКСПОРТ ДАННЫХ В EXCEL (раскомментировать при необходимости)
        # print("\n" + "=" * 60)
        # print("ЭКСПОРТ ДАННЫХ В EXCEL")
        # print("=" * 60)
        # DataExporter.export_comparison_to_excel(experiments_data, "all_experiments.xlsx")

    except Exception as e:
        print(f"Ошибка в основном цикле: {e}")
        import traceback
        traceback.print_exc()

    # ФИНАЛЬНЫЙ ОТЧЕТ
    _print_final_report(experiments_data)


def _print_final_report(experiments_data: Dict):
    """
    Печать финального отчета.
    """
    print("\n" + "=" * 60)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("=" * 60)

    print(f"Успешно выполнено экспериментов: {len(experiments_data)}")

    # Группируем по типам
    original_count = len([name for name in experiments_data.keys() if "оригинальный" in name])
    scipy_count = len([name for name in experiments_data.keys() if "SciPy" in name])

    no_obstacle_count = len([name for name in experiments_data.keys() if "без помех" in name])
    parallelepiped_count = len([name for name in experiments_data.keys() if "параллелепипедом" in name])
    cylinder_count = len([name for name in experiments_data.keys() if "цилиндрической помехой" in name])
    multiple_parallelepiped_count = len([name for name in experiments_data.keys() if "4 параллелепипедами" in name])

    print(f"Алгоритмы: оригинальный - {original_count}, SciPy - {scipy_count}")
    print(
        f"Помехи: без помех - {no_obstacle_count}, параллелепипед - {parallelepiped_count}, "
        f"цилиндр - {cylinder_count}, 4 параллелепипеда - {multiple_parallelepiped_count}")

    print("\nСозданные графики в папке 'plots/':")

    # Основные графики для каждого эксперимента
    for exp_name in experiments_data.keys():
        safe_name = exp_name.replace(" ", "_")
        print(f"\n--- {exp_name} ---")
        print(f"  main_curves_{safe_name}.png")
        print(f"  measurement_errors_{safe_name}.png")
        print(f"  smoothing_process_{safe_name}.png")
        print(f"  half_nodes_errors_{safe_name}.png")
        print(f"  derivative_analysis_{safe_name}.png")

    # Сравнительные графики
    print("\n--- СРАВНИТЕЛЬНЫЕ ГРАФИКИ ---")
    print("  comparison_main_curves.png")

    print("\nРЕКОМЕНДАЦИИ:")
    print("1. Для просмотра всех экспериментов: откройте comparison_main_curves.png")
    print("2. Для детального анализа: используйте индивидуальные графики экспериментов")
    print("3. Для экспорта данных в Excel: раскомментируйте код в main.py")
    print("=" * 60)


def build():
    # Инъекция метода из основной программы
    # level, spline0, spline = bspline_(data_series['raw_cumulative'])
    pass


if __name__ == "__main__":
    # Запуск полного набора экспериментов
    main()