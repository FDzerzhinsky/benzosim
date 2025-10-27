# visualization.py
"""
Визуализация результатов эксперимента с цилиндром в соответствии с оригинальным Pascal-кодом.
Включает только те графики, которые соответствуют выводу оригинальной программы.
"""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from analysis import AnalysisResults
from typing import Dict, List, Tuple, Any


class ExperimentVisualizer:
    """
    Класс для создания визуализаций результатов эксперимента в соответствии с Pascal-кодом.
    """

    def __init__(self, results: AnalysisResults):
        """
        Инициализация визуализатора.

        Args:
            results: Результаты анализа эксперимента
        """
        self.results = results

    def create_comprehensive_plots(self, H_ideal: np.ndarray, V_ideal: np.ndarray,
                                   Hs: np.ndarray, Vs: np.ndarray, Vs_smooth: np.ndarray,
                                   experiment_name: str = ""):
        """
        Алиас для обратной совместимости.
        Вызывает метод create_plots_according_to_pascal.
        """
        return self.create_plots_according_to_pascal(H_ideal, V_ideal, Hs, Vs, Vs_smooth, experiment_name)
    def create_plots_according_to_pascal(self, H_ideal: np.ndarray, V_ideal: np.ndarray,
                                         Hs: np.ndarray, Vs: np.ndarray, Vs_smooth: np.ndarray,
                                         experiment_name: str = ""):
        """
        Создание графиков в соответствии с оригинальным Pascal-кодом.

        Args:
            H_ideal: Идеальные значения высот
            V_ideal: Идеальные значения объемов
            Hs: Зашумленные значения высот
            Vs: Зашумленные значения объемов
            Vs_smooth: Сглаженные значения объемов
            experiment_name: Название эксперимента для подписей
        """
        # Создаем папку для графиков если её нет
        os.makedirs('plots', exist_ok=True)

        print(f"\nСоздание графиков для эксперимента: {experiment_name}")

        # 1. ОСНОВНЫЕ КРИВЫЕ - идеальная, зашумленная и сглаженная (аналог вывода в Pascal)
        self._plot_main_curves(H_ideal, V_ideal, Hs, Vs, Vs_smooth, experiment_name)

        # 2. ОШИБКИ ИЗМЕРЕНИЙ - по высоте и объему (аналог вывода D_H и Del_V в Pascal)
        self._plot_measurement_errors(H_ideal, V_ideal, Hs, Vs, experiment_name)

        # 3. ПРОЦЕСС СГЛАЖИВАНИЯ - итерационная сходимость (аналог вывода k, D_Max_V, Dev_Max_V в Pascal)
        if hasattr(self.results, 'iteration_stats') and len(self.results.iteration_stats) > 0:
            self._plot_smoothing_process(experiment_name)

        # 4. ПОЛУЦЕЛЫЕ УЗЛЫ - точность интерполяции (аналог вывода полуцелых узлов в Pascal)
        if hasattr(self.results, 'half_nodes_data') and len(self.results.half_nodes_data) > 0:
            self._plot_half_nodes_analysis(experiment_name)

        # 5. АНАЛИЗ ПРОИЗВОДНЫХ - ошибки производных (аналог вывода dV, dVs, dVss, Del_V1, Del_V2 в Pascal)
        if (hasattr(self.results, 'derivative_stats') and
                len(self.results.derivative_stats.get('Del_V1', [])) > 0):
            self._plot_derivative_analysis(Hs, experiment_name)

        print(f"Графики для '{experiment_name}' сохранены в папке 'plots/'")

    def _plot_main_curves(self, H_ideal: np.ndarray, V_ideal: np.ndarray,
                          Hs: np.ndarray, Vs: np.ndarray, Vs_smooth: np.ndarray,
                          experiment_name: str):
        """
        Основные кривые: идеальная, зашумленная и сглаженная.
        Соответствует выводу H, V(H), Vs, Vs_s в Pascal.
        """
        plt.figure(figsize=(15, 8))

        # Основные кривые
        plt.plot(H_ideal, V_ideal, 'b-', linewidth=2, label='Идеальная кривая V(H)')
        plt.plot(Hs, Vs, 'ro', markersize=2, alpha=0.6, label='Зашумленные данные (Vs)')
        plt.plot(Hs, Vs_smooth, 'g-', linewidth=1.5, label='Сглаженные данные (Vs_s)')

        # Области помех и анализа
        obstacle_info = self.results.geometry.get_obstacle_info()
        if obstacle_info:
            if obstacle_info['type'] == 'parallelepiped':
                plt.axvspan(obstacle_info['start_height'], obstacle_info['end_height'],
                            alpha=0.2, color='orange', label='Область параллелепипеда')
            elif obstacle_info['type'] == 'cylinder':
                plt.axvspan(obstacle_info['start_height'], obstacle_info['end_height'],
                            alpha=0.2, color='red', label='Область цилиндрической помехи')

        # Область анализа
        plt.axvspan(H_ideal[self.results.config.I_beg], H_ideal[self.results.config.I_end],
                    alpha=0.2, color='yellow', label='Область анализа (I_beg-I_end)')

        plt.xlabel('Высота H (см)')
        plt.ylabel('Объем V (л)')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.title(f'Основные кривые {experiment_name}')
        plt.tight_layout()
        plt.savefig(f'plots/main_curves_{experiment_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_measurement_errors(self, H_ideal: np.ndarray, V_ideal: np.ndarray,
                                 Hs: np.ndarray, Vs: np.ndarray, experiment_name: str):
        """
        Графики ошибок измерений по высоте и объему.
        Соответствует выводу D_H и Del_V в Pascal.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Ошибки по высоте (D_H)
        dH = H_ideal - Hs
        ax1.plot(H_ideal, dH, 'b-', linewidth=1)
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Ошибка высоты D_H (см)')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Ошибки измерений')

        # Статистика ошибок высоты
        mean_dH = np.mean(dH)
        max_dH = np.max(np.abs(dH))
        ax1.text(0.02, 0.98, f'Среднее: {mean_dH:.4f} см\nМакс: {max_dH:.4f} см',
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Относительные ошибки по объему (Del_V)
        dV_relative = (V_ideal - Vs) / V_ideal * 100
        ax2.plot(H_ideal, dV_relative, 'r-', linewidth=1)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Высота H (см)')
        ax2.set_ylabel('Относительная ошибка объема Del_V (%)')
        ax2.grid(True, alpha=0.3)

        # Статистика ошибок объема
        mean_dV = np.mean(dV_relative)
        max_dV = np.max(np.abs(dV_relative))
        ax2.text(0.02, 0.98, f'Среднее: {mean_dV:.4f}%\nМакс: {max_dV:.4f}%',
                 transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'plots/measurement_errors_{experiment_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_smoothing_process(self, experiment_name: str):
        """
        Визуализация процесса сглаживания - итерационная сходимость.
        Соответствует выводу k, D_Max_V, Dev_Max_V в Pascal.
        """
        # Проверяем, есть ли данные для построения графика
        if not hasattr(self.results, 'iteration_stats') or not self.results.iteration_stats:
            print(f"Предупреждение: Нет данных о процессе сглаживания для {experiment_name}")
            return

        # Проверяем, достаточно ли итераций для построения графика
        if len(self.results.iteration_stats) <= 1:
            print(
                f"Предупреждение: Недостаточно итераций для графика сглаживания ({len(self.results.iteration_stats)})")
            return

        plt.figure(figsize=(12, 6))

        iterations = [stat['iteration'] for stat in self.results.iteration_stats]
        max_errors = [stat['max_error'] for stat in self.results.iteration_stats]
        deriv_errors = [stat['derivative_error'] for stat in self.results.iteration_stats]

        plt.plot(iterations, max_errors, 'b-', linewidth=2, label='D_Max_V - макс ошибка объема')
        plt.plot(iterations, deriv_errors, 'r--', linewidth=2, label='Dev_Max_V - ошибка производной')

        # Целевая точность
        target_error = self.results.config.Del_V * 100
        plt.axhline(y=target_error, color='g', linestyle=':', alpha=0.7,
                    label=f'Целевая точность ({target_error:.2f}%)')

        plt.xlabel('Итерация сглаживания')
        plt.ylabel('Ошибка (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f'Процесс сглаживания данных - {experiment_name}')
        plt.savefig(f'plots/smoothing_process_{experiment_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"График процесса сглаживания создан: {len(iterations)} итераций")

    def _plot_half_nodes_analysis(self, experiment_name: str):
        """
        Анализ точности в полуцелых узлах.
        Соответствует выводу полуцелых узлов в Pascal.
        """
        plt.figure(figsize=(12, 6))

        half_nodes = self.results.half_nodes_data
        H_half = [node['H'] for node in half_nodes]
        errors1 = [node['error1'] for node in half_nodes]
        errors2 = [node['error2'] for node in half_nodes]

        plt.plot(H_half, errors1, 'b-', linewidth=1.5, label='Ошибка без коррекции (error1)')
        plt.plot(H_half, errors2, 'r--', linewidth=1.5, label='Ошибка с коррекцией (error2)')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        plt.xlabel('Высота в полуцелых узлах (см)')
        plt.ylabel('Ошибка в полуцелых узлах (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Точность интерполяции в промежуточных точках')
        plt.savefig(f'plots/half_nodes_errors_{experiment_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_derivative_analysis(self, Hs: np.ndarray, experiment_name: str):
        """
        Анализ ошибок производных.
        Соответствует выводу dV, dVs, dVss, Del_V1, Del_V2 в Pascal.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Используем правильные индексы для H_mid
        H_mid = Hs[self.results.config.I_beg:self.results.config.I_end + 1]

        # Ошибки производных
        ax1.plot(H_mid, self.results.derivative_stats['Del_V1'], 'b-',
                 label='Del_V1 - ошибка производной исходных данных')
        ax1.plot(H_mid, self.results.derivative_stats['Del_V2'], 'r--',
                 label='Del_V2 - ошибка производной сглаженных данных')
        ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Ошибка производной (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Ошибки производных объема')

        # Значения производных
        ax2.plot(H_mid, self.results.derivative_stats['dV_ideal'], 'b-',
                 label='dV - идеальная производная')
        ax2.plot(H_mid, self.results.derivative_stats['dVs'], 'r--',
                 label='dVs - производная зашумленных данных')
        ax2.plot(H_mid, self.results.derivative_stats['dVss'], 'g:',
                 label='dVss - производная сглаженных данных')
        ax2.set_xlabel('Высота H (см)')
        ax2.set_ylabel('Производная объема (л/см)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'plots/derivative_analysis_{experiment_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_comparison_plots(self, experiments_data: Dict[str, tuple], experiment_objects: Dict[str, Any]):
        """
        Сравнительные графики для нескольких экспериментов.
        Строит графики для всех комбинаций алгоритмов и типов помех.
        """
        print("\nСоздание сравнительных графиков...")

        # Сравнение основных кривых для всех экспериментов
        self._create_comparison_main_curves(experiments_data, experiment_objects)

        print("Сравнительные графики сохранены в 'plots/comparison_main_curves.png'")

    def _create_comparison_main_curves(self, experiments_data: Dict[str, tuple], experiment_objects: Dict[str, Any]):
        """
        Сравнение основных кривых для разных экспериментов.
        Располагает графики в матрице 3x3: строки - алгоритмы, столбцы - типы помех.
        """
        # Определяем порядок алгоритмов и типов помех
        algorithms_order = [
            "SciPy",
            "оригинальный Sg_p=1",
            "оригинальный Sg_p=2"
        ]

        obstacles_order = [
            "без помех",
            "с параллелепипедом",
            "с цилиндрической помехой"
        ]

        # Создаем фигуру 3x3
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

        # Проходим по всем комбинациям алгоритмов и помех
        for row_idx, algorithm in enumerate(algorithms_order):
            for col_idx, obstacle in enumerate(obstacles_order):
                ax = axes[row_idx, col_idx]

                # Формируем название эксперимента для поиска
                target_name = f"{obstacle} - {algorithm}"

                # Ищем эксперимент с таким названием
                found_data = None
                found_experiment = None

                for exp_name, data in experiments_data.items():
                    if target_name in exp_name:
                        found_data = data
                        if exp_name in experiment_objects:
                            found_experiment = experiment_objects[exp_name]
                        break

                if found_data is not None:
                    H_ideal, V_ideal, Hs, Vs, Vs_smooth = found_data
                    color = colors[(row_idx * 3 + col_idx) % len(colors)]

                    # Основные кривые
                    ax.plot(H_ideal, V_ideal, 'k-', linewidth=1, label='Идеальная V(H)')
                    ax.plot(Hs, Vs_smooth, '-', linewidth=1.5, color=color, label=target_name)

                    # Область анализа
                    ax.axvspan(H_ideal[self.results.config.I_beg], H_ideal[self.results.config.I_end],
                               alpha=0.2, color='yellow', label='Область анализа')

                    # Области помех
                    if found_experiment:
                        obstacle_info = found_experiment.geometry.get_obstacle_info()
                        if obstacle_info:
                            if obstacle_info['type'] == 'parallelepiped':
                                ax.axvspan(obstacle_info['start_height'], obstacle_info['end_height'],
                                           alpha=0.3, color='orange', label='Параллелепипед')
                            elif obstacle_info['type'] == 'cylinder':
                                ax.axvspan(obstacle_info['start_height'], obstacle_info['end_height'],
                                           alpha=0.3, color='red', label='Цилиндрическая помеха')

                    # Настройки графика
                    ax.set_title(f'{target_name}', fontsize=10)
                    ax.set_xlabel('Высота H (см)')
                    ax.set_ylabel('Объем V (л)')
                    ax.legend(fontsize=7)
                    ax.grid(True, alpha=0.3)
                else:
                    # Если эксперимент не найден, скрываем ось
                    ax.set_visible(False)

        # Добавляем общие подписи для строк и столбцов
        for row_idx, algorithm in enumerate(algorithms_order):
            axes[row_idx, 0].set_ylabel(f'{algorithm}\nОбъем V (л)', fontsize=12, fontweight='bold')

        for col_idx, obstacle in enumerate(obstacles_order):
            axes[0, col_idx].set_title(f'{obstacle}', fontsize=12, fontweight='bold')

        # Общий заголовок
        fig.suptitle('Сравнение алгоритмов сглаживания для различных типов помех',
                     fontsize=16, fontweight='bold', y=0.95)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig('plots/comparison_main_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Сравнительные графики сохранены в 'plots/comparison_main_curves.png'")


class DataExporter:
    """
    Класс для выгрузки данных в формате, соответствующем оригинальному Pascal-коду.
    """

    @staticmethod
    def export_to_excel(results: AnalysisResults, H_ideal: np.ndarray, V_ideal: np.ndarray,
                        Hs: np.ndarray, Vs: np.ndarray, Vs_smooth: np.ndarray,
                        filename: str = "results.xlsx"):
        """
        Экспорт данных в Excel файл с заголовками как в оригинальном Pascal-коде.

        Args:
            results: Результаты анализа
            H_ideal: Идеальные высоты
            V_ideal: Идеальные объемы
            Hs: Зашумленные высоты
            Vs: Зашумленные объемы
            Vs_smooth: Сглаженные объемы
            filename: Имя файла для сохранения
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:

            # 1. Основные данные (H, V, Hs, Vs, D_H, Del_V)
            main_data = []
            for i in range(len(H_ideal)):
                dH = H_ideal[i] - Hs[i]
                dV_rel = (V_ideal[i] - Vs[i]) / (V_ideal[i] + 1e-10) * 100

                main_data.append({
                    'H_см': H_ideal[i],
                    'V_л': V_ideal[i],
                    'Hs_см': Hs[i],
                    'Vs_л': Vs[i],
                    'D_H_см': dH,
                    'Del_V_%': dV_rel
                })

            df_main = pd.DataFrame(main_data)
            df_main.to_excel(writer, sheet_name='Основные данные', index=False)

            # 2. Данные сглаживания
            if hasattr(results, 'smoothing_comparison') and results.smoothing_comparison:
                smooth_data = []
                for item in results.smoothing_comparison:
                    smooth_data.append({
                        'H_см': item['H'],
                        'V_л': item['V_ideal'],
                        'Vs_л': item['Vs'],
                        'Vs_s_л': item['Vs_smooth']
                    })

                df_smooth = pd.DataFrame(smooth_data)
                df_smooth.to_excel(writer, sheet_name='Сглаживание', index=False)

            # 3. Полуцелые узлы
            if hasattr(results, 'half_nodes_data') and results.half_nodes_data:
                half_nodes_data = []
                for item in results.half_nodes_data:
                    half_nodes_data.append({
                        'H_полуцелое_см': item['H'],
                        'V_идеальное_л': item['V_ideal'],
                        'V_сглаженное_л': item['V_smooth'],
                        'V_сглаженное_корр_л': item['V_smooth_3'],
                        'error1_%': item['error1'],
                        'error2_%': item['error2']
                    })

                df_half = pd.DataFrame(half_nodes_data)
                df_half.to_excel(writer, sheet_name='Полуцелые узлы', index=False)

            # 4. Производные
            if (hasattr(results, 'derivative_stats') and
                    results.derivative_stats.get('Del_V1')):

                deriv_data = []
                working_range = slice(results.config.I_beg, results.config.I_end + 1)
                H_mid = Hs[working_range]

                for i in range(len(results.derivative_stats['Del_V1'])):
                    deriv_data.append({
                        'Hs_см': H_mid[i] if i < len(H_mid) else 0,
                        'dV_л': results.derivative_stats['dV_ideal'][i] if i < len(
                            results.derivative_stats['dV_ideal']) else 0,
                        'dVs_л': results.derivative_stats['dVs'][i] if i < len(results.derivative_stats['dVs']) else 0,
                        'dVss_л': results.derivative_stats['dVss'][i] if i < len(
                            results.derivative_stats['dVss']) else 0,
                        'Del_V1_%': results.derivative_stats['Del_V1'][i] if i < len(
                            results.derivative_stats['Del_V1']) else 0,
                        'Del_V2_%': results.derivative_stats['Del_V2'][i] if i < len(
                            results.derivative_stats['Del_V2']) else 0
                    })

                df_deriv = pd.DataFrame(deriv_data)
                df_deriv.to_excel(writer, sheet_name='Производные', index=False)

            # 5. Статистика ошибок
            error_stats = []
            if hasattr(results, 'error_stats'):
                error_stats.append({
                    'Min_dH_см': results.error_stats.get('min_dH', 0),
                    'Max_dH_см': results.error_stats.get('max_dH', 0),
                    'Min_dV_%': results.error_stats.get('min_dV', 0),
                    'Max_dV_%': results.error_stats.get('max_dV', 0)
                })

            if hasattr(results, 'derivative_stats'):
                error_stats[0].update({
                    'Min_Del_V1_%': results.derivative_stats.get('min_Del_V1', 0),
                    'Max_Del_V1_%': results.derivative_stats.get('max_Del_V1', 0),
                    'Min_Del_V2_%': results.derivative_stats.get('min_Del_V2', 0),
                    'Max_Del_V2_%': results.derivative_stats.get('max_Del_V2', 0),
                    'S_dV_%': results.derivative_stats.get('mean_Del_V1', 0)
                })

            df_stats = pd.DataFrame(error_stats)
            df_stats.to_excel(writer, sheet_name='Статистика', index=False)

            # 6. Процесс сглаживания
            if hasattr(results, 'iteration_stats') and results.iteration_stats:
                iter_data = []
                for stat in results.iteration_stats:
                    iter_data.append({
                        'k': stat['iteration'],
                        'D_Max_V_%': stat['max_error'],
                        'Dev_Max_V_%': stat['derivative_error'],
                        'c_M': stat['c_M'],
                        'i_cM': stat['i_cM'],
                        'i_cv0': stat['i_cv0']
                    })

                df_iter = pd.DataFrame(iter_data)
                df_iter.to_excel(writer, sheet_name='Итерации сглаживания', index=False)

        print(f"Данные экспортированы в файл: {filename}")

    @staticmethod
    def export_comparison_to_excel(experiments_data: Dict[str, tuple], filename: str = "comparison.xlsx"):
        """
        Экспорт сравнительных данных всех экспериментов.

        Args:
            experiments_data: Словарь с данными экспериментов
            filename: Имя файла для сохранения
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:

            for exp_name, data in experiments_data.items():
                H_ideal, V_ideal, Hs, Vs, Vs_smooth = data

                # Создаем лист для каждого эксперимента
                exp_data = []
                for i in range(len(H_ideal)):
                    dH = H_ideal[i] - Hs[i]
                    dV_rel = (V_ideal[i] - Vs[i]) / (V_ideal[i] + 1e-10) * 100
                    dV_smooth_rel = (V_ideal[i] - Vs_smooth[i]) / (V_ideal[i] + 1e-10) * 100

                    exp_data.append({
                        'H_см': H_ideal[i],
                        'V_идеальное_л': V_ideal[i],
                        'Hs_см': Hs[i],
                        'Vs_зашумленное_л': Vs[i],
                        'Vs_сглаженное_л': Vs_smooth[i],
                        'D_H_см': dH,
                        'Del_V_зашумленное_%': dV_rel,
                        'Del_V_сглаженное_%': dV_smooth_rel
                    })

                df_exp = pd.DataFrame(exp_data)
                # Ограничиваем имя листа до 31 символа (ограничение Excel)
                sheet_name = exp_name[:31] if len(exp_name) > 31 else exp_name
                df_exp.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Сравнительные данные экспортированы в файл: {filename}")