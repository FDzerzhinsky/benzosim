# experiment.py
"""
Основной класс эксперимента с поддержкой двух алгоритмов сглаживания.
Обеспечивает полную совместимость с оригинальным Pascal-выводом.
"""

from geometry import CylinderGeometry
from data_processing import DataGenerator, SmoothingProcessor
from analysis import AnalysisResults
from visualization import ExperimentVisualizer
from config import ExperimentConfig, ObstacleType


class CylinderExperiment:
    """
    Класс эксперимента с цилиндром.
    Объединяет все компоненты: геометрию, генерацию данных, сглаживание, анализ и визуализацию.
    """

    def __init__(self, config: ExperimentConfig, name: str = ""):
        """
        Инициализация эксперимента.

        Args:
            config: Конфигурация эксперимента
            name: Название эксперимента для идентификации
        """
        self.config = config
        self.name = name

        # Инициализация компонентов эксперимента
        self.geometry = CylinderGeometry(config)
        self.data_generator = DataGenerator(self.geometry, config)

        # Инициализация процессора сглаживания с выбранным алгоритмом
        self.smoothing_processor = SmoothingProcessor(config, config.use_original_smoothing)

        self.results = AnalysisResults(config, self.geometry)
        self.visualizer = ExperimentVisualizer(self.results)

        # Данные эксперимента
        self.H_ideal = None
        self.V_ideal = None
        self.Hs = None
        self.Vs = None
        self.Vs_smooth = None

    def run_experiment(self):
        """
        Запуск полного эксперимента с выводом в формате Pascal.

        Returns:
            Кортеж с данными эксперимента или None в случае ошибки
        """
        try:
            # Вывод информации о конфигурации
            self.config.print_config()

            # Формируем информацию о типе помехи для вывода
            obstacle_info = ""
            if self.config.obstacle_type == ObstacleType.PARALLELEPIPED:
                obstacle_info = f" (параллелепипед: h={self.config.parallelepiped_height}см, w={self.config.parallelepiped_width}см)"
            elif self.config.obstacle_type == ObstacleType.CYLINDER:
                obstacle_info = f" (цилиндр: h={self.config.cylinder_height}см, r={self.config.cylinder_radius}см)"

            # Вывод заголовка как в Pascal
            print(
                f" R={self.config.R:6.2f} см, L={self.config.L:6.2f} см, Vb={self.geometry.Vb:7.4f} л  Kn={self.config.Kn:3} Sg_p={self.config.Sg_p:2}{obstacle_info}")

            # 1. ГЕНЕРАЦИЯ ИДЕАЛЬНЫХ ДАННЫХ
            print("\n      H      V(H)")
            self.H_ideal, self.V_ideal = self.data_generator.generate_ideal_data()

            # Вывод первых точек (как в Pascal)
            for i in range(min(10, len(self.H_ideal))):
                print(f"  {self.H_ideal[i]:6.2f}  {self.V_ideal[i]:8.2f}")
            if len(self.H_ideal) > 10:
                print("  ...")
            print()

            # 2. ГЕНЕРАЦИЯ ЗАШУМЛЕННЫХ ДАННЫХ
            self.Hs, self.Vs = self.data_generator.add_measurement_noise(self.H_ideal, self.V_ideal)

            print("   H      Hs         V          Vs")
            for i in range(1, min(20, len(self.H_ideal) - 1)):
                print(f" {self.H_ideal[i]:5.2f}  {self.Hs[i]:6.3f}  {self.V_ideal[i]:9.3f}  {self.Vs[i]:9.3f}")
            if len(self.H_ideal) > 20:
                print("  ...")
            print()

            # 3. СГЛАЖИВАНИЕ ДАННЫХ (выбранным алгоритмом)
            print(f"Начало сглаживания... use_original_smoothing={self.config.use_original_smoothing}")
            self.Vs_smooth, iteration_stats = self.smoothing_processor.smooth_data(
                self.H_ideal, self.V_ideal, self.Vs)

            # Сохраняем статистику итераций
            self.results.iteration_stats = iteration_stats
            print(f"Сглаживание завершено. Итераций: {len(iteration_stats)}")

            # Отладочная информация
            if iteration_stats:
                print("Статистика итераций сохранена:")
                for stat in iteration_stats:
                    print(f"  Итерация {stat['iteration']}: ошибка={stat['max_error']:.4f}%")
            else:
                print("Предупреждение: статистика итераций пуста")

            # 4. РАСЧЕТ ПОЛУЦЕЛЫХ УЗЛОВ
            self.results.calculate_half_nodes(self.H_ideal, self.V_ideal, self.Vs_smooth)

            # 5. ВЫВОД РЕЗУЛЬТАТОВ СГЛАЖИВАНИЯ
            print("\n  Сглаживание ")
            print("   H       V        Vs     Vs_s")

            # Создаем данные для вывода сглаживания
            self.results.smoothing_comparison = []
            for i in range(self.config.I_beg, self.config.I_end + 1):
                self.results.smoothing_comparison.append({
                    'H': self.H_ideal[i],
                    'V_ideal': self.V_ideal[i],
                    'Vs': self.Vs[i],
                    'Vs_smooth': self.Vs_smooth[i]
                })

            # Вывод первых точек
            comparison_count = min(10, len(self.results.smoothing_comparison))
            for i in range(comparison_count):
                data = self.results.smoothing_comparison[i]
                print(f" {data['H']:5.1f}  {data['V_ideal']:9.3f}  {data['Vs']:9.3f}  {data['Vs_smooth']:9.3f}")
            if len(self.results.smoothing_comparison) > 10:
                print("  ...")
            print()

            # 6. ДЕТАЛЬНОЕ СРАВНЕНИЕ И РАСЧЕТ ОШИБОК
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

            # 7. СТАТИСТИКИ ОШИБОК
            print(f" I_beg={self.config.I_beg:4}  I_end={self.config.I_end:4}")
            print(
                f" Min_dH={self.results.error_stats['min_dH']:8.4f} (см)  Max_dH={self.results.error_stats['max_dH']:8.4f} "
                f"(см)   Min_dV={self.results.error_stats['min_dV']:8.4f} (%)  Max_dV={self.results.error_stats['max_dV']:8.4f} (%)")
            print()

            # 8. АНАЛИЗ ПРОИЗВОДНЫХ
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

            # 9. ИТОГОВЫЕ СТАТИСТИКИ
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

            # ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
            self.visualizer.create_plots_according_to_pascal(
                self.H_ideal, self.V_ideal, self.Hs, self.Vs, self.Vs_smooth, self.name)

            return self.H_ideal, self.V_ideal, self.Hs, self.Vs, self.Vs_smooth

        except Exception as e:
            print(f"Ошибка в эксперименте {self.name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_data(self):
        """
        Получение данных эксперимента.

        Returns:
            Кортеж с всеми массивами данных
        """
        return self.H_ideal, self.V_ideal, self.Hs, self.Vs, self.Vs_smooth