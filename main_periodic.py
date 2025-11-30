# main_periodic.py
"""
Главный скрипт для запуска периодических экспериментов
Использует готовые конфигурации из config.py
"""

from periodic_experiment import PeriodicDrainingExperiment
from config import PERIODIC_CONFIGS


def main():
    """Запуск всех периодических экспериментов с конфигами из config.py"""
    print("ПЕРИОДИЧЕСКИЕ ЭКСПЕРИМЕНТЫ С ЦИЛИНДРОМ")
    print("=" * 60)
    print("Используются конфигурации из config.py")
    print("=" * 60)

    # Запускаем все конфигурации из словаря PERIODIC_CONFIGS
    for config_name, config in PERIODIC_CONFIGS.items():
        experiment_name = f"Периодический - {config.name}"
        print(f"\n{'-' * 60}")
        print(f"ЗАПУСК: {experiment_name}")
        print(f"Конфиг: {config_name}")
        print(f"{'-' * 60}")

        experiment = PeriodicDrainingExperiment(config, experiment_name)
        experiment.run_experiment()

    print("\n" + "=" * 60)
    print("ВСЕ ПЕРИОДИЧЕСКИЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("=" * 60)


if __name__ == "__main__":
    main()