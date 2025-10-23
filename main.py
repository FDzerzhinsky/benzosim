from experiment import CylinderExperiment
from visualization import ExperimentVisualizer
from config import ExperimentConfig
from analysis import AnalysisResults
from geometry import CylinderGeometry


def main():
    print("ЭКСПЕРИМЕНТ С ЦИЛИНДРОМ НА БОКУ - ПОЛНАЯ РЕАЛИЗАЦИЯ PASCAL ПРОГРАММЫ")
    print("=" * 80)

    experiments_data = {}

    try:
        # Эксперимент с параллелепипедом
        print("\n" + "=" * 80)
        print("ЭКСПЕРИМЕНТ С ПАРАЛЛЕЛЕПИПЕДОМ")
        print("=" * 80)

        config_with = ExperimentConfig(use_parallelepiped=True)
        experiment_with = CylinderExperiment(config_with, "с параллелепипедом")
        data_with = experiment_with.run_experiment()

        if data_with:
            experiments_data["с параллелепипедом"] = data_with

        # Эксперимент без параллелепипеда
        print("\n" + "=" * 80)
        print("ЭКСПЕРИМЕНТ БЕЗ ПАРАЛЛЕЛЕПИПЕДА")
        print("=" * 80)

        config_without = ExperimentConfig(use_parallelepiped=False)
        experiment_without = CylinderExperiment(config_without, "без параллелепипеда")
        data_without = experiment_without.run_experiment()

        if data_without:
            experiments_data["без параллелепипеда"] = data_without

        # Сравнительная визуализация
        if len(experiments_data) >= 2:
            visualizer = ExperimentVisualizer(experiment_with.results)
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
    print("  - main_curves_с параллелепипедом.png (основные кривые)")
    print("  - main_curves_без параллелепипеда.png (основные кривые)")
    print("  - measurement_errors_...png (ошибки измерений)")
    print("  - smoothing_process_...png (процесс сглаживания)")
    print("  - half_nodes_errors_...png (полуцелые узлы)")
    print("  - derivative_analysis_...png (производные)")
    if len(experiments_data) >= 2:
        print("  - parallelepiped_comparison.png (сравнение экспериментов)")
    print("=" * 80)


if __name__ == "__main__":
    main()