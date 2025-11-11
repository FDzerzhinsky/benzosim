# config.py
"""
Конфигурация эксперимента с параметрами из оригинального Pascal-кода.
Добавлен параметр выбора алгоритма сглаживания.
"""

from dataclasses import dataclass
import os
from enum import Enum


class ObstacleType(Enum):
    """Типы помех в цилинре"""
    NONE = "none"
    PARALLELEPIPED = "parallelepiped"
    CYLINDER = "cylinder"
    MULTIPLE_PARALLELEPIPEDS = "multiple_parallelepipeds"  # НОВЫЙ ТИП


@dataclass
class ExperimentConfig:
    """
    Конфигурация эксперимента - точное соответствие параметрам Pascal-кода.
    Все параметры имеют значения по умолчанию из оригинальной программы.
    """

    # Основные геометрические параметры (как в Pascal)
    R: float = 100.0  # Радиус цилиндра [см]
    L: float = 500.0  # Длина цилиндра [см]
    Nf: int = 200  # Количество точек по высоте
    dh: float = 1.0  # Шаг по высоте [см]

    # Параметры шума измерений (как в Pascal)
    Del_h: float = 0.05  # Амплитуда шума по высоте [см]
    Del_V: float = 0.0015  # Относительная амплитуда шума по объему [доли]

    # Параметры алгоритма (как в Pascal)
    Kn: int = 1  # Количество реализаций (повторений эксперимента)
    I_beg: int = 50  # Начальный индекс рабочего диапазона
    I_end: int = 150  # Конечный индекс рабочего диапазона
    I_p: int = 10  # Дополнительные точки для сглаживания
    Ro: float = 0.035  # Параметр сглаживания
    Sg_p: int = 1  # Параметр метода сглаживания: 1 - конечные разности, 2 - сплайны

    # Параметры параллелепипеда (помеха)
    parallelepiped_height: float = 60.0  # Высота нижней грани параллелепипеда [см]
    parallelepiped_width: float = 10.0  # Ширина параллелепипеда [см]
    parallelepiped_length: float = 500.0  # Длина параллелепипеда [см]

    # Параметры цилиндрической помехи
    cylinder_radius: float = 5.0  # Радиус цилиндрической помехи [см]
    cylinder_height: float = 65.0  # Высота нижней точки цилиндра [см]
    cylinder_length: float = 500.0  # Длина цилиндрической помехи [см]

    # НОВЫЕ ПАРАМЕТРЫ ДЛЯ МНОГИХ ПАРАЛЛЕЛЕПИПЕДОВ
    multiple_parallelepiped_heights: list = None  # Список высот нижних граней
    multiple_parallelepiped_widths: list = None  # Список ширин
    multiple_parallelepiped_lengths: list = None  # Список длин

    # Тип помехи (по умолчанию - без помех)
    obstacle_type: ObstacleType = ObstacleType.NONE

    # Параметры генерации случайных чисел
    seed: int = 42  # Seed для детерминированных случайных чисел

    # НОВЫЙ ПАРАМЕТР: выбор алгоритма сглаживания
    use_original_smoothing: bool = False  # False = SciPy, True = оригинальный алгоритм из Pascal

    def __post_init__(self):
        """
        Пост-инициализация: создание необходимых директорий.
        Вызывается автоматически после __init__.
        """
        os.makedirs('plots', exist_ok=True)  # Создание папки для графиков

        # Инициализация списков для многих параллелепипедов
        if self.multiple_parallelepiped_heights is None:
            self.multiple_parallelepiped_heights = [30.0, 60.0, 90.0, 120.0]
        if self.multiple_parallelepiped_widths is None:
            self.multiple_parallelepiped_widths = [8.0, 8.0, 8.0, 8.0]
        if self.multiple_parallelepiped_lengths is None:
            self.multiple_parallelepiped_lengths = [500.0, 500.0, 500.0, 500.0]

    def print_config(self):
        """Вывод конфигурации в удобочитаемом формате"""
        print("=" * 60)
        print("КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА")
        print("=" * 60)
        print(f"Геометрия: R={self.R} см, L={self.L} см")
        print(f"Точки: Nf={self.Nf}, шаг dh={self.dh} см")
        print(f"Шум: Del_h={self.Del_h} см, Del_V={self.Del_V * 100:.2f}%")
        print(f"Диапазон анализа: I_beg={self.I_beg}, I_end={self.I_end}")
        print(f"Алгоритм: Ro={self.Ro}, Sg_p={self.Sg_p}")
        print(f"Тип помехи: {self.obstacle_type.value}")

        if self.obstacle_type == ObstacleType.PARALLELEPIPED:
            print(f"Параллелепипед: h={self.parallelepiped_height} см, w={self.parallelepiped_width} см")
        elif self.obstacle_type == ObstacleType.CYLINDER:
            print(f"Цилиндр: h={self.cylinder_height} см, r={self.cylinder_radius} см")
        elif self.obstacle_type == ObstacleType.MULTIPLE_PARALLELEPIPEDS:
            print(f"Количество параллелепипедов: {len(self.multiple_parallelepiped_heights)}")
            for i, (h, w) in enumerate(zip(self.multiple_parallelepiped_heights, self.multiple_parallelepiped_widths)):
                print(f"  Параллелепипед {i + 1}: h={h} см, w={w} см")

        algorithm = "ОРИГИНАЛЬНЫЙ (Pascal)" if self.use_original_smoothing else "SciPy"
        print(f"Алгоритм сглаживания: {algorithm}")
        print("=" * 60)