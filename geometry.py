# geometry.py
"""
Геометрические расчеты для цилиндра на боку с учетом различных типов помех.
Точная реализация функций V_h и dV_h из оригинального Pascal-кода.
"""

import math
import numpy as np
from config import ExperimentConfig, ObstacleType


class CylinderGeometry:
    """
    Класс для геометрических расчетов цилиндра, лежащего на боку.
    Включает расчет объемов и производных с учетом помех различного типа.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Инициализация геометрии цилиндра.

        Args:
            config: Конфигурация эксперимента с параметрами цилиндра и помех
        """
        self.config = config
        # Полный объем цилиндра в литрах (аналог Vb в Pascal)
        self.Vb = math.pi * config.L * config.R * config.R / 1e3

    def V_h(self, h: float) -> float:
        """
        Вычисление нормированного объема V(h) как в Pascal-функции V_h.
        Точное воспроизведение логики оригинального кода.

        Args:
            h: Безразмерная высота (h/R), где h - уровень жидкости от дна

        Returns:
            Нормированный объем (без умножения на R*R*L)
        """
        # Расчет объема сегмента круга (как в Pascal)
        if h <= 1.0:
            # Уровень жидкости ниже центра
            alfa = 2 * math.acos(1.0 - h - 1.0e-14)  # Малый сдвиг для избежания деления на 0
            V = 0.5 * (alfa - math.sin(alfa))
        else:
            # Уровень жидкости выше центра
            alfa = 2 * math.acos(h - 1.0 - 1.0e-14)  # Малый сдвиг для избежания деления на 0
            V = 0.5 * (2 * math.pi - alfa + math.sin(alfa))

        # Обработка различных типов помех (точное соответствие Pascal)
        if self.config.obstacle_type == ObstacleType.PARALLELEPIPED:
            # Нормированные параметры параллелепипеда
            ha_norm = self.config.parallelepiped_height / self.config.R
            a_norm = self.config.parallelepiped_width / self.config.R

            # Вычитание объема, занятого параллелепипедом
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
        """
        Производная dV/dh - точная реализация Pascal-функции dV_h.

        Args:
            h: Безразмерная высота (h/R)

        Returns:
            Производная нормированного объема по безразмерной высоте
        """
        # Расчет производной для основного объема (как в Pascal)
        if h <= 1.0:
            alfa = 2 * math.acos(1.0 - h - 1.0e-14)
        else:
            alfa = 2 * math.acos(h - 1.0 - 1.0e-14)

        # Производная d(alfa)/dh
        dAlfa_h = 2 / math.sqrt(1 - (1 - h) ** 2 + 1.0e-14)

        # Основная производная объема
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
                    # Производная для нижней половины цилиндра
                    dV_cyl = math.sqrt(2 * r_cyl_norm * h_cyl - h_cyl * h_cyl)
                else:
                    # Производная для верхней половины цилиндра
                    h_cyl_remaining = 2 * r_cyl_norm - h_cyl
                    dV_cyl = -math.sqrt(2 * r_cyl_norm * h_cyl_remaining - h_cyl_remaining * h_cyl_remaining)

                dV = dV - r_cyl_norm * r_cyl_norm * dV_cyl

        return dV

    def calculate_ideal_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Расчет идеальной кривой V(H) без шума.

        Returns:
            H: Массив высот от 0 до 2R с шагом dh [см]
            V: Массив соответствующих объемов [л]
        """
        # Создание массива высот (как в Pascal: от 0 до Nf*dh)
        H = np.arange(0, self.config.Nf + 1) * self.config.dh

        # Расчет объемов для каждой высоты
        V = np.array([self.config.R * self.config.R * self.config.L *
                      self.V_h(h / self.config.R) / 1e3 for h in H])

        return H, V

    def get_obstacle_info(self) -> dict or None:
        """
        Получение информации о помехе для визуализации.

        Returns:
            Словарь с информацией о помехе или None если помех нет
        """
        if self.config.obstacle_type == ObstacleType.NONE:
            return None

        elif self.config.obstacle_type == ObstacleType.PARALLELEPIPED:
            return {
                'type': 'parallelepiped',
                'start_height': self.config.parallelepiped_height,
                'end_height': self.config.parallelepiped_height + self.config.parallelepiped_width,
                'volume_at_start': self.V_h(
                    self.config.parallelepiped_height / self.config.R) *
                                   self.config.R * self.config.R * self.config.L / 1e3,
                'volume_at_end': self.V_h(
                    (self.config.parallelepiped_height + self.config.parallelepiped_width) /
                    self.config.R) * self.config.R * self.config.R * self.config.L / 1e3
            }

        elif self.config.obstacle_type == ObstacleType.CYLINDER:
            return {
                'type': 'cylinder',
                'start_height': self.config.cylinder_height,
                'end_height': self.config.cylinder_height + 2 * self.config.cylinder_radius,
                'radius': self.config.cylinder_radius,
                'volume_at_start': self.V_h(
                    self.config.cylinder_height / self.config.R) *
                                   self.config.R * self.config.R * self.config.L / 1e3,
                'volume_at_end': self.V_h(
                    (self.config.cylinder_height + 2 * self.config.cylinder_radius) /
                    self.config.R) * self.config.R * self.config.R * self.config.L / 1e3
            }

    def calculate_segment_volume(self, h: float, R: float) -> float:
        """
        Вычисление объема сегмента круга (вспомогательная функция).

        Args:
            h: Высота сегмента [см]
            R: Радиус цилиндра [см]

        Returns:
            Объем сегмента [см³]
        """
        if h <= R:
            # Нижний сегмент
            alpha = 2 * math.acos((R - h) / R)
            return R ** 2 / 2 * (alpha - math.sin(alpha))
        else:
            # Верхний сегмент (объем цилиндра минус нижний сегмент)
            alpha = 2 * math.acos((h - R) / R)
            return math.pi * R ** 2 - R ** 2 / 2 * (alpha - math.sin(alpha))

    def print_geometry_info(self):
        """Вывод информации о геометрии цилиндра"""
        print("=" * 50)
        print("ИНФОРМАЦИЯ О ГЕОМЕТРИИ ЦИЛИНДРА")
        print("=" * 50)
        print(f"Радиус цилиндра (R): {self.config.R} см")
        print(f"Длина цилиндра (L): {self.config.L} см")
        print(f"Полный объем (Vb): {self.Vb:.2f} л")
        print(f"Диаметр цилиндра: {2 * self.config.R} см")

        obstacle_info = self.get_obstacle_info()
        if obstacle_info:
            print(f"Тип помехи: {obstacle_info['type']}")
            if obstacle_info['type'] == 'parallelepiped':
                print(f"  Высота: {obstacle_info['start_height']} см")
                print(f"  Ширина: {obstacle_info['end_height'] - obstacle_info['start_height']} см")
            elif obstacle_info['type'] == 'cylinder':
                print(f"  Высота центра: {obstacle_info['start_height']} см")
                print(f"  Радиус: {obstacle_info['radius']} см")
        else:
            print("Помехи: отсутствуют")
        print("=" * 50)