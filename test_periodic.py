# test_periodic.py
"""
Тестовый скрипт для проверки периодического эксперимента
Использует готовые конфигурации из config.py
"""

from periodic_experiment import PeriodicDrainingExperiment
from config import PERIODIC_TEST_CONFIG, QUICK_TEST_CONFIG


def test_periodic_experiment():
    """Тестовый запуск периодического эксперимента с конфигом из config.py"""
    print("ТЕСТОВЫЙ ЗАПУСК ПЕРИОДИЧЕСКОГО ЭКСПЕРИМЕНТА")

    # Используем готовую конфигурацию из config.py
    experiment = PeriodicDrainingExperiment(
        PERIODIC_TEST_CONFIG,
        "Тестовый эксперимент (конфиг из config.py)"
    )
    experiment.run_experiment()

    print("\nТЕСТ ЗАВЕРШЕН!")


def test_quick_experiment():
    """Быстрый тест с другой конфигурацией"""
    print("\n" + "=" * 50)
    print("БЫСТРЫЙ ТЕСТ")
    print("=" * 50)

    experiment = PeriodicDrainingExperiment(
        QUICK_TEST_CONFIG,
        "Быстрый тестовый эксперимент"
    )
    experiment.run_experiment()


if __name__ == "__main__":
    test_periodic_experiment()
    # test_quick_experiment()  # Раскомментировать для дополнительного теста