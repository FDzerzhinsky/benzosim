# main.py
from experiment import CylinderExperiment
from visualization import ExperimentVisualizer
from config import ExperimentConfig, ObstacleType


def main():
    print("ЭКСПЕРИМЕНТ С ЦИЛИНДРОМ НА БОКУ - РАСШИРЕННАЯ ВЕРСИЯ С РАЗЛИЧНЫМИ ПОМЕХАМИ")
    print("=" * 80)

    experiments_data = {}

    try:
        # Эксперимент без помех
        print("\n" + "=" * 80)
        print("ЭКСПЕРИМЕНТ БЕЗ ПОМЕХ")
        print("=" * 80)

        config_none = ExperimentConfig(obstacle_type=ObstacleType.NONE)
        experiment_none = CylinderExperiment(config_none, "без помех")
        data_none = experiment_none.run_experiment()

        if data_none:
            experiments_data["без помех"] = data_none

        # Эксперимент с параллелепипедом
        print("\n" + "=" * 80)
        print("ЭКСПЕРИМЕНТ С ПАРАЛЛЕЛЕПИПЕДОМ")
        print("=" * 80)

        config_parallelepiped = ExperimentConfig(obstacle_type=ObstacleType.PARALLELEPIPED)
        experiment_parallelepiped = CylinderExperiment(config_parallelepiped, "с параллелепипедом")
        data_parallelepiped = experiment_parallelepiped.run_experiment()

        if data_parallelepiped:
            experiments_data["с параллелепипедом"] = data_parallelepiped

        # Эксперимент с цилиндрической помехой (используются значения по умолчанию из конфига)
        print("\n" + "=" * 80)
        print("ЭКСПЕРИМЕНТ С ЦИЛИНДРИЧЕСКОЙ ПОМЕХОЙ")
        print("=" * 80)

        # Используем значения по умолчанию из конфига, которые вы изменили
        config_cylinder = ExperimentConfig(obstacle_type=ObstacleType.CYLINDER)
        experiment_cylinder = CylinderExperiment(config_cylinder, "с цилиндрической помехой")
        data_cylinder = experiment_cylinder.run_experiment()

        if data_cylinder:
            experiments_data["с цилиндрической помехой"] = data_cylinder

        # Сравнительная визуализация
        if len(experiments_data) >= 2:
            visualizer = ExperimentVisualizer(experiment_none.results)
            visualizer.create_comparison_plots(experiments_data)
        else:
            print("Недостаточно данных для сравнения экспериментов")

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("=" * 80)
    print("\nСозданные файлы в папке 'plots/':")
    for exp_name in experiments_data.keys():
        print(f"  - main_curves_{exp_name}.png (основные кривые)")
        print(f"  - measurement_errors_{exp_name}.png (ошибки измерений)")
        print(f"  - smoothing_process_{exp_name}.png (процесс сглаживания)")
        print(f"  - half_nodes_errors_{exp_name}.png (полуцелые узлы)")
        print(f"  - derivative_analysis_{exp_name}.png (производные)")
    if len(experiments_data) >= 2:
        print("  - obstacle_comparison.png (сравнение экспериментов)")
    print("=" * 80)


if __name__ == "__main__":
    main()