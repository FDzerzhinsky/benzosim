from dataclasses import dataclass
import os


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
    ha: float = 60.0
    a: float = 10.0
    seed: int = 42
    use_parallelepiped: bool = True

    def __post_init__(self):
        # Создаем папку для графиков
        os.makedirs('plots', exist_ok=True)