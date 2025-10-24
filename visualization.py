# visualization.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from analysis import AnalysisResults


class ExperimentVisualizer:
    def __init__(self, results: AnalysisResults):
        self.results = results

    def create_comprehensive_plots(self, H_ideal: np.ndarray, V_ideal: np.ndarray,
                                   Hs: np.ndarray, Vs: np.ndarray, Vs_smooth: np.ndarray,
                                   experiment_name: str = ""):

        plt.figure(figsize=(19.2, 10.8))
        plt.plot(H_ideal, V_ideal, 'b-', linewidth=2, label='Идеальная кривая V(H)')
        plt.plot(Hs, Vs, 'ro', markersize=2, alpha=0.6, label='Зашумленные данные Vs')
        plt.plot(Hs, Vs_smooth, 'g-', linewidth=1.5, label='Сглаженные данные Vs_s')

        # Используем новый метод get_obstacle_info() вместо get_parallelepiped_info()
        obstacle_info = self.results.geometry.get_obstacle_info()
        if obstacle_info:
            if obstacle_info['type'] == 'parallelepiped':
                plt.axvspan(obstacle_info['start_height'], obstacle_info['end_height'],
                            alpha=0.2, color='orange', label='Область параллелепипеда')
            elif obstacle_info['type'] == 'cylinder':
                plt.axvspan(obstacle_info['start_height'], obstacle_info['end_height'],
                            alpha=0.2, color='red', label='Область цилиндрической помехи')

        plt.axvspan(H_ideal[self.results.config.I_beg], H_ideal[self.results.config.I_end],
                    alpha=0.2, color='yellow', label='Область анализа I_beg-I_end')

        plt.xlabel('Высота H (см)')
        plt.ylabel('Объем V (л)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f'Основные кривые {experiment_name}')
        plt.savefig(f'plots/main_curves_{experiment_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. График ошибок
        plt.figure(figsize=(19.2, 10.8))

        plt.subplot(2, 1, 1)
        dH = H_ideal - Hs
        plt.plot(H_ideal, dH, 'b-', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.ylabel('Ошибка высоты D_H (см)')
        plt.grid(True, alpha=0.3)
        plt.title('Ошибки измерений')

        plt.subplot(2, 1, 2)
        dV_relative = (V_ideal - Vs) / V_ideal * 100
        plt.plot(H_ideal, dV_relative, 'r-', linewidth=1)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Высота H (см)')
        plt.ylabel('Относительная ошибка объема Del_V (%)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'plots/measurement_errors_{experiment_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Процесс сглаживания
        if len(self.results.iteration_stats) > 0:
            plt.figure(figsize=(19.2, 10.8))
            iterations = [stat['iteration'] for stat in self.results.iteration_stats]
            max_errors = [stat['max_error'] for stat in self.results.iteration_stats]
            deriv_errors = [stat['derivative_error'] for stat in self.results.iteration_stats]

            plt.plot(iterations, max_errors, 'b-', linewidth=2, label='D_Max_V - макс ошибка объема')
            plt.plot(iterations, deriv_errors, 'r--', linewidth=2, label='Dev_Max_V - ошибка производной')
            plt.xlabel('Итерация сглаживания')
            plt.ylabel('Ошибка (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Процесс сглаживания данных')
            plt.savefig(f'plots/smoothing_process_{experiment_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Полуцелые узлы
        if len(self.results.half_nodes_data) > 0:
            plt.figure(figsize=(19.2, 10.8))
            half_nodes = self.results.half_nodes_data
            H_half = [node['H'] for node in half_nodes]
            errors = [node['error1'] for node in half_nodes]
            plt.plot(H_half, errors, 'g-', linewidth=1)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Высота в полуцелых узлах (см)')
            plt.ylabel('Ошибка в полуцелых узлах (%)')
            plt.grid(True, alpha=0.3)
            plt.title('Точность интерполяции в промежуточных точках')
            plt.savefig(f'plots/half_nodes_errors_{experiment_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 5. Ошибки производных
        if len(self.results.derivative_stats['Del_V1']) > 0:
            plt.figure(figsize=(19.2, 10.8))

            # Используем правильные индексы для H_mid
            H_mid = self.results.Hs[self.results.config.I_beg:self.results.config.I_end + 1]

            plt.subplot(2, 1, 1)
            plt.plot(H_mid, self.results.derivative_stats['Del_V1'], 'b-',
                     label='Del_V1 - ошибка производной исходных данных')
            plt.plot(H_mid, self.results.derivative_stats['Del_V2'], 'r--',
                     label='Del_V2 - ошибка производной сглаженных данных')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.ylabel('Ошибка производной (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Ошибки производных объема')

            plt.subplot(2, 1, 2)
            plt.plot(H_mid, self.results.derivative_stats['dV_ideal'], 'b-', label='dV - идеальная производная')
            plt.plot(H_mid, self.results.derivative_stats['dVs'], 'r--', label='dVs - производная зашумленных данных')
            plt.plot(H_mid, self.results.derivative_stats['dVss'], 'g:', label='dVss - производная сглаженных данных')
            plt.xlabel('Высота H (см)')
            plt.ylabel('Производная объема (л/см)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'plots/derivative_analysis_{experiment_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Графики для '{experiment_name}' сохранены в папке 'plots/'")

    def create_comparison_plots(self, experiments_data: dict):
        """Сравнение экспериментов с различными типами помех"""
        fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))

        for idx, (exp_name, data) in enumerate(experiments_data.items()):
            ax = axes[idx // 2, idx % 2]

            H_ideal, V_ideal, Hs, Vs, Vs_smooth = data
            ax.plot(H_ideal, V_ideal, 'b-', linewidth=2, label='Идеальная V(H)')
            ax.plot(Hs, Vs, 'ro', markersize=1, alpha=0.6, label='Зашумленная Vs')
            ax.plot(Hs, Vs_smooth, 'g-', linewidth=1, label='Сглаженная Vs_s')

            # Используем новый метод get_obstacle_info() для определения типа помехи
            obstacle_info = self.results.geometry.get_obstacle_info()
            if obstacle_info:
                if obstacle_info['type'] == 'parallelepiped' and "параллелепипед" in exp_name:
                    ax.axvspan(obstacle_info['start_height'], obstacle_info['end_height'],
                               alpha=0.3, color='orange', label='Параллелепипед')
                elif obstacle_info['type'] == 'cylinder' and "цилиндр" in exp_name:
                    ax.axvspan(obstacle_info['start_height'], obstacle_info['end_height'],
                               alpha=0.3, color='red', label='Цилиндрическая помеха')

            ax.set_title(f'Эксперимент: {exp_name}')
            ax.set_xlabel('Высота H (см)')
            ax.set_ylabel('Объем V (л)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('plots/obstacle_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Сравнение экспериментов сохранено в 'plots/obstacle_comparison.png'")