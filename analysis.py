import numpy as np
from typing import List, Dict
from config import ExperimentConfig
from geometry import CylinderGeometry


class AnalysisResults:
    def __init__(self, config: ExperimentConfig, geometry: CylinderGeometry):
        self.config = config
        self.geometry = geometry
        self.half_nodes_data = []
        self.smoothing_comparison = []
        self.error_stats = {}
        self.derivative_stats = {}
        self.iteration_stats = []
        self.detailed_comparison = []
        self.Hs = None  # Добавим для хранения Hs

    def calculate_half_nodes(self, H: np.ndarray, V_ideal: np.ndarray, V_smooth: np.ndarray):
        """Расчет данных в полуцелых узлах - КАК В PASCAL"""
        print("\nПолуцелые узлы ")

        for i in range(self.config.I_beg, self.config.I_end):
            H_half = H[i] + 0.5
            V_ideal_half = self.geometry.config.R * self.geometry.config.R * self.geometry.config.L * \
                           self.geometry.V_h(H_half / self.geometry.config.R) / 1e3
            V_smooth_half = (V_smooth[i] + V_smooth[i + 1]) / 2
            # Упрощение для ошибки 2
            V_smooth_half_3 = V_smooth_half - 0.001  # небольшая коррекция как в Pascal

            error1 = (1 - V_smooth_half / V_ideal_half) * 100
            error2 = (1 - V_smooth_half_3 / V_ideal_half) * 100

            self.half_nodes_data.append({
                'H': H_half,
                'V_ideal': V_ideal_half,
                'V_smooth': V_smooth_half,
                'V_smooth_3': V_smooth_half_3,
                'error1': error1,
                'error2': error2
            })

            print(f"   {H_half:7.3f} {V_ideal_half:9.3f}  {V_smooth_half:9.3f}  "
                  f"{V_smooth_half_3:9.3f}  {error1:9.3f}  {error2:9.3f}")

    def calculate_errors(self, H_ideal: np.ndarray, V_ideal: np.ndarray,
                         Hs: np.ndarray, Vs: np.ndarray, Vs_smooth: np.ndarray):
        """Расчет всех ошибок и статистик - КАК В PASCAL"""
        self.Hs = Hs  # Сохраняем Hs для визуализации

        # Ошибки по высоте и объему
        dH = H_ideal - Hs
        dV_relative = (V_ideal - Vs) / (V_ideal + 1e-10) * 100

        # Статистики в рабочем диапазоне
        mask = slice(self.config.I_beg, self.config.I_end + 1)

        Max_dH = 0
        Min_dH = 0
        Max_dV = 0
        Min_dV = 0

        for i in range(self.config.I_beg, self.config.I_end + 1):
            if dH[i] > Max_dH:
                Max_dH = dH[i]
            if dH[i] < Min_dH:
                Min_dH = dH[i]
            if dV_relative[i] > Max_dV:
                Max_dV = dV_relative[i]
            if dV_relative[i] < Min_dV:
                Min_dV = dV_relative[i]

        self.error_stats = {
            'min_dH': Min_dH,
            'max_dH': Max_dH,
            'min_dV': Min_dV,
            'max_dV': Max_dV
        }

        # Ошибки производных
        dV_ideal = []
        dVs = []
        dVss = []
        Del_V1_arr = []
        Del_V2_arr = []

        for i in range(self.config.I_beg, self.config.I_end + 1):
            dV_i = (V_ideal[i] - V_ideal[i - 1]) / 10  # как в Pascal
            dVs_i = (Vs[i] - Vs[i - 1]) / 10
            # Исправление: проверка деления на ноль
            h_diff = Hs[i] - Hs[i - 1]
            if abs(h_diff) < 1e-10:
                h_diff = 1.0  # избегаем деления на ноль
            dVss_i = (Vs_smooth[i] - Vs_smooth[i - 1]) / h_diff / 10

            Del_V1 = ((dV_i - dVs_i) / (dV_i + 1e-10)) * 100
            Del_V2 = ((dV_i - dVss_i) / (dV_i + 1e-10)) * 100

            dV_ideal.append(dV_i)
            dVs.append(dVs_i)
            dVss.append(dVss_i)
            Del_V1_arr.append(Del_V1)
            Del_V2_arr.append(Del_V2)

        # Статистики производных
        Max_Del_V1 = max(Del_V1_arr) if Del_V1_arr else 0
        Min_Del_V1 = min(Del_V1_arr) if Del_V1_arr else 0
        Max_Del_V2 = max(Del_V2_arr) if Del_V2_arr else 0
        Min_Del_V2 = min(Del_V2_arr) if Del_V2_arr else 0
        S_dV = sum(Del_V1_arr) / len(Del_V1_arr) if Del_V1_arr else 0

        self.derivative_stats = {
            'dV_ideal': dV_ideal,
            'dVs': dVs,
            'dVss': dVss,
            'Del_V1': Del_V1_arr,
            'Del_V2': Del_V2_arr,
            'min_Del_V1': Min_Del_V1,
            'max_Del_V1': Max_Del_V1,
            'min_Del_V2': Min_Del_V2,
            'max_Del_V2': Max_Del_V2,
            'mean_Del_V1': S_dV
        }

        # Сохранение для детального вывода
        self.smoothing_comparison = []
        self.detailed_comparison = []

        for i in range(self.config.I_beg, self.config.I_end + 1):
            self.smoothing_comparison.append({
                'H': H_ideal[i],
                'V_ideal': V_ideal[i],
                'Vs': Vs[i],
                'Vs_smooth': Vs_smooth[i]
            })

        # Детальное сравнение для вывода как в Pascal
        for i in range(len(H_ideal)):
            dH_val = H_ideal[i] - Hs[i]
            dV_rel = (V_ideal[i] - Vs[i]) / (V_ideal[i] + 1e-10) * 100

            self.detailed_comparison.append({
                'H_ideal': H_ideal[i],
                'V_ideal': V_ideal[i],
                'Hs': Hs[i],
                'Vs': Vs[i],
                'dH': dH_val,
                'dV_rel': dV_rel
            })