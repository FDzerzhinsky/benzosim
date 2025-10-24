# config.py
from dataclasses import dataclass
import os
from enum import Enum

class ObstacleType(Enum):
    NONE = "none"
    PARALLELEPIPED = "parallelepiped"
    CYLINDER = "cylinder"

@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента"""
    R: float = 100.0
    L: float = 500.0
    Nf: int = 200
    dh: float = 1.0
    Del_h: float = 0.05
    Del_V: float = 0.0015
    Kn: int = 1
    I_beg: int = 50
    I_end: int = 150
    I_p: int = 10
    Ro: float = 0.035
    Sg_p: int = 1
    # Параметры параллелепипеда
    parallelepiped_height: float = 60.0
    parallelepiped_width: float = 10.0
    parallelepiped_length: float = 500.0
    # Параметры цилиндрической помехи - уменьшен размер и смещен ниже
    cylinder_radius: float = 10.0  # Уменьшен с 30.0 до 15.0 см
    cylinder_height: float = 65.0  #
    cylinder_length: float = 500.0
    # Тип помехи
    obstacle_type: ObstacleType = ObstacleType.NONE
    seed: int = 42

    def __post_init__(self):
        os.makedirs('plots', exist_ok=True)