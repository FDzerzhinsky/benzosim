# experiment.py
from geometry import CylinderGeometry
from data_processing import DataGenerator, SmoothingProcessor
from analysis import AnalysisResults
from visualization import ExperimentVisualizer
from config import ExperimentConfig, ObstacleType


class CylinderExperiment:
    def __init__(self, config: ExperimentConfig, name: str = ""):
        self.config = config
        self.name = name
        self.geometry = CylinderGeometry(config)
        self.data_generator = DataGenerator(self.geometry, config)
        self.smoothing_processor = SmoothingProcessor(config)
        self.results = AnalysisResults(config, self.geometry)
        self.visualizer = ExperimentVisualizer(self.results)

        self.H_ideal = None
        self.V_ideal = None
        self.Hs = None
        self.Vs = None
        self.Vs_smooth = None

    def run_experiment(self):
        """Запуск эксперимента с выводом КАК В PASCAL"""
        try:
            # Формируем информацию о типе помехи для вывода
            obstacle_info = ""
            if self.config.obstacle_type == ObstacleType.PARALLELEPIPED:
                obstacle_info = f" (параллелепипед: h={self.config.parallelepiped_height}см, w={self.config.parallelepiped_width}см)"
            elif self.config.obstacle_type == ObstacleType.CYLINDER:
                obstacle_info = f" (цилиндр: h={self.config.cylinder_height}см, r={self.config.cylinder_radius}см)"

            print(
                f" R={self.config.R:6.2f} см, L={self.config.L:6.2f} см, Vb={self.geometry.Vb:7.4f} л  Kn={self.config.Kn:3} Sg_p={self.config.Sg_p:2}{obstacle_info}")

            # 1. Генерация идеальных данных
            print("\n      H      V(H)")
            self.H_ideal, self.V_ideal = self.data_generator.generate_ideal_data()
            for i in range(min(10, len(self.H_ideal))):  # Ограничим вывод
                print(f"  {self.H_ideal[i]:6.2f}  {self.V_ideal[i]:8.2f}")
            if len(self.H_ideal) > 10:
                print("  ...")
            print()

            # 2. Генерация зашумленных данных
            self.Hs, self.Vs = self.data_generator.add_measurement_noise(self.H_ideal, self.V_ideal)

            print("   H      Hs         V          Vs")
            for i in range(1, min(20, len(self.H_ideal) - 1)):  # Выводим первые 20 точек
                print(f" {self.H_ideal[i]:5.2f}  {self.Hs[i]:6.3f}  {self.V_ideal[i]:9.3f}  {self.Vs[i]:9.3f}")
            if len(self.H_ideal) > 20:
                print("  ...")
            print()

            # 3. Сглаживание данных
            self.Vs_smooth, iteration_stats = self.smoothing_processor.smooth_data(
                self.H_ideal, self.V_ideal, self.Vs)
            self.results.iteration_stats = iteration_stats

            # 4. Полуцелые узлы
            self.results.calculate_half_nodes(self.H_ideal, self.V_ideal, self.Vs_smooth)

            # 5. Вывод сглаживания
            print("\n  Сглаживание ")
            print("   H       V        Vs     Vs_s")
            comparison_count = min(10, len(self.results.smoothing_comparison))
            for i in range(comparison_count):
                data = self.results.smoothing_comparison[i]
                print(f" {data['H']:5.1f}  {data['V_ideal']:9.3f}  {data['Vs']:9.3f}  {data['Vs_smooth']:9.3f}")
            if len(self.results.smoothing_comparison) > 10:
                print("  ...")
            print()

            # 6. Детальное сравнение
            self.results.calculate_errors(self.H_ideal, self.V_ideal, self.Hs, self.Vs, self.Vs_smooth)

            print("     H(см)      V(л)       Hs(см)      Vs(л)    D_H(см)  Del_V(%)")
            for i in range(min(10, len(self.results.detailed_comparison))):
                data = self.results.detailed_comparison[i]
                print(f"  {data['H_ideal']:7.3f}  {data['V_ideal']:10.3f}    "
                      f"{data['Hs']:7.3f}  {data['Vs']:10.3f}    "
                      f"{data['dH']:7.4f} {data['dV_rel']:8.3f}")
            if len(self.results.detailed_comparison) > 10:
                print("  ...")
            print()

            # 7. Статистики ошибок
            print(f" I_beg={self.config.I_beg:4}  I_end={self.config.I_end:4}")
            print(
                f" Min_dH={self.results.error_stats['min_dH']:8.4f} (см)  Max_dH={self.results.error_stats['max_dH']:8.4f} "
                f"(см)   Min_dV={self.results.error_stats['min_dV']:8.4f} (%)  Max_dV={self.results.error_stats['max_dV']:8.4f} (%)")
            print()

            # 8. Производные
            print("  Hs(см)   dV(л)    dVs(л)  dVss(л)   Del_V1%  Del_V2%")
            derivative_count = min(10, len(self.results.derivative_stats['Del_V1']))
            for i in range(derivative_count):
                print(f"  {self.Hs[self.config.I_beg + i]:5.1f}  "
                      f"{self.results.derivative_stats['dV_ideal'][i]:7.3f}  "
                      f"{self.results.derivative_stats['dVs'][i]:7.3f}  "
                      f"{self.results.derivative_stats['dVss'][i]:7.3f}   "
                      f"{self.results.derivative_stats['Del_V1'][i]:7.3f}  "
                      f"{self.results.derivative_stats['Del_V2'][i]:7.3f}")
            if len(self.results.derivative_stats['Del_V1']) > 10:
                print("  ...")
            print()

            # 9. Итоговые статистики
            print(
                f" Min_dH={self.results.error_stats['min_dH']:8.4f} (см)  Max_dH={self.results.error_stats['max_dH']:8.4f} "
                f"(см)   Min_dV={self.results.error_stats['min_dV']:8.4f} (%)  Max_dV={self.results.error_stats['max_dV']:8.4f} (%)")
            print(f" S_dV={self.results.derivative_stats['mean_Del_V1']:7.4f} "
                  f"Min_Del_V1={self.results.derivative_stats['min_Del_V1']:7.3f} (%)   "
                  f"Max_Del_V1={self.results.derivative_stats['max_Del_V1']:7.3f} (%)   "
                  f"Min_Del_V2={self.results.derivative_stats['min_Del_V2']:7.3f} (%)  "
                  f"Max_Del_V2={self.results.derivative_stats['max_Del_V2']:7.3f} (%)")

            print()
            print("  Stop")

            # Визуализация
            self.visualizer.create_comprehensive_plots(
                self.H_ideal, self.V_ideal, self.Hs, self.Vs, self.Vs_smooth, self.name)

            return self.H_ideal, self.V_ideal, self.Hs, self.Vs, self.Vs_smooth

        except Exception as e:
            print(f"Ошибка в эксперименте {self.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_data(self):
        return self.H_ideal, self.V_ideal, self.Hs, self.Vs, self.Vs_smooth