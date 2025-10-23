import math
import numpy as np
from config import ExperimentConfig


class CylinderGeometry:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.Vb = math.pi * config.L * config.R * config.R / 1e3

    def V_h(self, h: float) -> float:
        """Объем с учетом параллелепипеда - ТОЧНАЯ РЕАЛИЗАЦИЯ ИЗ PASCAL"""
        if h <= 1.0:
            alfa = 2 * math.acos(1.0 - h - 1.0e-14)
            V = 0.5 * (alfa - math.sin(alfa))
        else:
            alfa = 2 * math.acos(h - 1.0 - 1.0e-14)
            V = 0.5 * (2 * math.pi - alfa + math.sin(alfa))

        # Параллелепипед (только если включен в конфиге)
        if self.config.use_parallelepiped:
            ha_norm = self.config.ha / self.config.R
            a_norm = self.config.a / self.config.R

            if (ha_norm <= h) and (h <= ha_norm + a_norm):
                V = V - a_norm * (h - ha_norm)
            if h > ha_norm + a_norm:
                V = V - a_norm * a_norm

        return V

    def dV_h(self, h: float) -> float:
        """Производная dV/dh - ТОЧНАЯ РЕАЛИЗАЦИЯ ИЗ PASCAL"""
        if h <= 1.0:
            alfa = 2 * math.acos(1.0 - h - 1.0e-14)
        else:
            alfa = 2 * math.acos(h - 1.0 - 1.0e-14)

        dAlfa_h = 2 / math.sqrt(1 - (1 - h) ** 2 + 1.0e-14)
        dV = 0.5 * (1 - math.cos(alfa)) * (dAlfa_h + 1.0e-14)
        return dV

    def calculate_ideal_curve(self):
        """Расчет идеальной кривой V(H)"""
        H = np.arange(0, self.config.Nf + 1) * self.config.dh
        V = np.array([self.config.R * self.config.R * self.config.L *
                      self.V_h(h / self.config.R) / 1e3 for h in H])
        return H, V

    def get_parallelepiped_info(self):
        """Информация о параллелепипеде для визуализации"""
        if not self.config.use_parallelepiped:
            return None

        return {
            'start_height': self.config.ha,
            'end_height': self.config.ha + self.config.a,
            'volume_at_start': self.V_h(
                self.config.ha / self.config.R) * self.config.R * self.config.R * self.config.L / 1e3,
            'volume_at_end': self.V_h(
                (self.config.ha + self.config.a) / self.config.R) * self.config.R * self.config.R * self.config.L / 1e3
        }