# geometry.py
import math
import numpy as np
from config import ExperimentConfig, ObstacleType


class CylinderGeometry:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.Vb = math.pi * config.L * config.R * config.R / 1e3

    def V_h(self, h: float) -> float:
        """Объем с учетом различных типов помех"""
        if h <= 1.0:
            alfa = 2 * math.acos(1.0 - h - 1.0e-14)
            V = 0.5 * (alfa - math.sin(alfa))
        else:
            alfa = 2 * math.acos(h - 1.0 - 1.0e-14)
            V = 0.5 * (2 * math.pi - alfa + math.sin(alfa))

        # Обработка различных типов помех
        if self.config.obstacle_type == ObstacleType.PARALLELEPIPED:
            ha_norm = self.config.parallelepiped_height / self.config.R
            a_norm = self.config.parallelepiped_width / self.config.R

            if (ha_norm <= h) and (h <= ha_norm + a_norm):
                V = V - a_norm * (h - ha_norm)
            if h > ha_norm + a_norm:
                V = V - a_norm * a_norm

        elif self.config.obstacle_type == ObstacleType.CYLINDER:
            # Цилиндрическая помеха
            h_cyl_norm = self.config.cylinder_height / self.config.R
            r_cyl_norm = self.config.cylinder_radius / self.config.R

            # Если уровень жидкости выше нижней точки цилиндра
            if h >= h_cyl_norm:
                # Высота жидкости в цилиндрической помехе
                h_cyl = min(h - h_cyl_norm, 2 * r_cyl_norm)

                # Объем цилиндрической помехи, занятый жидкостью
                if h_cyl <= r_cyl_norm:
                    # Сегмент цилиндра при h_cyl <= r_cyl_norm
                    alpha_cyl = 2 * math.acos(1 - h_cyl / r_cyl_norm - 1.0e-14)
                    V_cyl_segment = 0.5 * (alpha_cyl - math.sin(alpha_cyl))
                else:
                    # Сегмент цилиндра при h_cyl > r_cyl_norm
                    h_cyl_remaining = 2 * r_cyl_norm - h_cyl
                    alpha_cyl = 2 * math.acos(1 - h_cyl_remaining / r_cyl_norm - 1.0e-14)
                    V_cyl_segment = math.pi - 0.5 * (alpha_cyl - math.sin(alpha_cyl))

                # Вычитаем объем, занятый цилиндрической помехой
                V = V - r_cyl_norm * r_cyl_norm * V_cyl_segment

        return V

    def dV_h(self, h: float) -> float:
        """Производная dV/dh"""
        if h <= 1.0:
            alfa = 2 * math.acos(1.0 - h - 1.0e-14)
        else:
            alfa = 2 * math.acos(h - 1.0 - 1.0e-14)

        dAlfa_h = 2 / math.sqrt(1 - (1 - h) ** 2 + 1.0e-14)
        dV = 0.5 * (1 - math.cos(alfa)) * (dAlfa_h + 1.0e-14)

        # Корректировка производной для различных типов помех
        if self.config.obstacle_type == ObstacleType.PARALLELEPIPED:
            ha_norm = self.config.parallelepiped_height / self.config.R
            a_norm = self.config.parallelepiped_width / self.config.R

            if (ha_norm <= h) and (h <= ha_norm + a_norm):
                dV = dV - a_norm

        elif self.config.obstacle_type == ObstacleType.CYLINDER:
            h_cyl_norm = self.config.cylinder_height / self.config.R
            r_cyl_norm = self.config.cylinder_radius / self.config.R

            if h >= h_cyl_norm and h <= h_cyl_norm + 2 * r_cyl_norm:
                h_cyl = h - h_cyl_norm
                if h_cyl <= r_cyl_norm:
                    dV_cyl = math.sqrt(2 * r_cyl_norm * h_cyl - h_cyl * h_cyl)
                else:
                    h_cyl_remaining = 2 * r_cyl_norm - h_cyl
                    dV_cyl = -math.sqrt(2 * r_cyl_norm * h_cyl_remaining - h_cyl_remaining * h_cyl_remaining)
                dV = dV - r_cyl_norm * r_cyl_norm * dV_cyl

        return dV

    def calculate_ideal_curve(self):
        """Расчет идеальной кривой V(H)"""
        H = np.arange(0, self.config.Nf + 1) * self.config.dh
        V = np.array([self.config.R * self.config.R * self.config.L *
                      self.V_h(h / self.config.R) / 1e3 for h in H])
        return H, V

    def get_obstacle_info(self):
        """Информация о помехе для визуализации"""
        if self.config.obstacle_type == ObstacleType.NONE:
            return None
        elif self.config.obstacle_type == ObstacleType.PARALLELEPIPED:
            return {
                'type': 'parallelepiped',
                'start_height': self.config.parallelepiped_height,
                'end_height': self.config.parallelepiped_height + self.config.parallelepiped_width,
                'volume_at_start': self.V_h(
                    self.config.parallelepiped_height / self.config.R) * self.config.R * self.config.R * self.config.L / 1e3,
                'volume_at_end': self.V_h(
                    (
                                self.config.parallelepiped_height + self.config.parallelepiped_width) / self.config.R) * self.config.R * self.config.R * self.config.L / 1e3
            }
        elif self.config.obstacle_type == ObstacleType.CYLINDER:
            return {
                'type': 'cylinder',
                'start_height': self.config.cylinder_height,
                'end_height': self.config.cylinder_height + 2 * self.config.cylinder_radius,
                'radius': self.config.cylinder_radius,
                'volume_at_start': self.V_h(
                    self.config.cylinder_height / self.config.R) * self.config.R * self.config.R * self.config.L / 1e3,
                'volume_at_end': self.V_h(
                    (
                                self.config.cylinder_height + 2 * self.config.cylinder_radius) / self.config.R) * self.config.R * self.config.R * self.config.L / 1e3
            }