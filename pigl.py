# pigl.py
"""
Минималистичный эксперимент с реальными данными.
Только загрузка, шум, сглаживание и графики.
"""

import numpy as np
import matplotlib

matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils import *


def main():
    print("=== ЭКСПЕРИМЕНТ С РЕАЛЬНЫМИ ДАННЫМИ ===")

    # =========================================================================
    # ФЛАГИ ВКЛЮЧЕНИЯ АЛГОРИТМОВ
    # =========================================================================
    USE_SCIPY = False  # Включить SciPy сглаживание
    USE_ORIG1 = False  # Включить оригинальный алгоритм Sg_p=1
    USE_ORIG2 = True  # Включить оригинальный алгоритм Sg_p=2

    # =========================================================================
    # ПАРАМЕТРЫ СГЛАЖИВАНИЯ - НАСТРОЙКИ АЛГОРИТМОВ
    # =========================================================================

    # Общие параметры для всех алгоритмов
    I_beg = 20  # Начальный индекс рабочего диапазона
    I_end = 240  # Конечный индекс рабочего диапазона (260-20=240)

    # Параметры шума (как в оригинальном Pascal-коде)
    Del_h = 0.05  # Амплитуда шума по высоте [см]
    Del_V = 0.0015  # Относительная амплитуда шума по объему [доли]

    # Параметры для оригинального алгоритма Sg_p=1 (конечные разности)
    params_orig1 = {
        'I_p': 8,  # Дополнительные точки для сглаживания
        'Ro': 0.05,  # Параметр сглаживания (консервативный)
        'max_iter': 40,  # Максимальное количество итераций
        'target_error': 0.0005  # Целевая точность (0.05%)

        # 'I_p': 8,  # Дополнительные точки для сглаживания
        # 'Ro': 0.02,  # Параметр сглаживания (консервативный)
        # 'max_iter': 30,  # Максимальное количество итераций
        # 'target_error': 0.0005  # Целевая точность (0.05%)
    }

    # Параметры для оригинального алгоритма Sg_p=2 (сплайны)
    params_orig2 = {
        'I_p': 10,  # Дополнительные точки для сглаживания
        'Ro': 0.043,  # Параметр сглаживания (консервативный)
        'max_iter': 40,  # Максимальное количество итераций
        'target_error': 0.0005  # Целевая точность (0.03%)

        # 'I_p': 10,  # Дополнительные точки для сглаживания
        # 'Ro': 0.015,  # Параметр сглаживания (консервативный)
        # 'max_iter': 40,  # Максимальное количество итераций
        # 'target_error': 0.0003  # Целевая точность (0.03%)
    }

    # Параметры для SciPy алгоритма
    scipy_smooth_factor = 0.1  # Параметр сглаживания для UnivariateSpline

    # =========================================================================
    # СТИЛИ ЛИНИЙ ДЛЯ ГРАФИКОВ
    # =========================================================================

    # Стили для разных типов данных
    line_styles = {
        'real_data': {'color': 'blue', 'linestyle': '-', 'linewidth': 3, 'label': 'Реальные данные (ИСХОДНЫЕ)'},
        'noisy_data': {'color': 'red', 'linestyle': ':', 'linewidth': 2, 'label': 'Зашумленные данные'},
        'scipy_smooth': {'color': 'brown', 'linestyle': '--', 'linewidth': 1, 'label': 'Сглаженные (SciPy)'},
        'orig1_smooth': {'color': 'orange', 'linestyle': '-.', 'linewidth': 1, 'label': 'Сглаженные (Ориг. Sg_p=1)'},
        'orig2_smooth': {'color': 'green', 'linestyle': '--', 'linewidth': 2, 'label': 'Сглаженные (Ориг. Sg_p=2)'}
    }

    print("ПАРАМЕТРЫ СГЛАЖИВАНИЯ:")
    print(f"  Рабочий диапазон: I_beg={I_beg}, I_end={I_end}")
    print(f"  Шум: Del_h={Del_h}, Del_V={Del_V}")
    print(f"  Оригинальный Sg_p=1: {params_orig1}")
    print(f"  Оригинальный Sg_p=2: {params_orig2}")
    print(f"  SciPy: smooth_factor={scipy_smooth_factor}")
    print()

    # 1. ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ
    print("1. Загрузка реальных данных...")
    loader = DataLoaderXLSX('data/53063_export.xlsx', ['storage_level'],
                            ['DATE_TIME', 'STOR_ID', 'OIL_LEVEL', 'VOLUME', 'DELTA'])
    data = loader.data

    rawdata = data['storage_level']
    storages = list(set(rawdata['STOR_ID'].tolist()))
    data_storages = {}
    for stor_id in storages:
        data_storages[stor_id] = rawdata[rawdata['STOR_ID'] == stor_id]

    # Берем первый резервуар
    stor_id = storages[0]
    df = data_storages[stor_id]

    # ТОЛЬКО загрузка, без сортировок и проверок
    H_real = df['OIL_LEVEL'].values
    V_real = df['VOLUME'].values

    # ПРЕДВАРИТЕЛЬНОЕ СГЛАЖИВАНИЕ
    PRE_SMOOTH = False  # Флаг включения/выключения предварительного сглаживания
    PRE_SMOOTH_METHOD = "original"  # "scipy" или "original"

    if PRE_SMOOTH:
        print("Предварительное сглаживание исходных данных...")

        if PRE_SMOOTH_METHOD == "scipy":
            # SciPy сглаживание
            pre_smooth_factor = 0.01
            V_real = scipy_smooth(H_real, V_real, I_beg=0, I_end=len(H_real) - 1,
                                  smooth_factor=pre_smooth_factor)
            print(f"   Метод: SciPy, factor={pre_smooth_factor}")

        elif PRE_SMOOTH_METHOD == "original":
            # Безопасный оригинальный алгоритм
            pre_smooth_params = {
                'I_p': 5,
                'Ro': 0.02,
                'max_iter': 20,
                'target_error': 0.001,
                'Sg_p': 2  # Теперь можно использовать Sg_p=2
            }
            safe_smoother = SafeOriginalSmoother()
            V_real = safe_smoother.safe_smooth_data(H_real, V_real, **pre_smooth_params)
            print(f"   Метод: SafeOriginal, params={pre_smooth_params}")

        print("   Предварительное сглаживание завершено")


    print(f"   Загружено: {len(H_real)} точек")
    print(f"   Высота: {H_real.min():.1f} - {H_real.max():.1f} см")
    print(f"   Объем: {V_real.min():.1f} - {V_real.max():.1f} л")

    # 2. РАСЧЕТ ПРОИЗВОДНОЙ РЕАЛЬНЫХ ДАННЫХ
    print("2. Расчет производной реальных данных...")
    dV_real = calculate_derivative(H_real, V_real)

    # 3. ВНЕСЕНИЕ ВОЗМУЩЕНИЙ
    print("3. Внесение возмущений...")
    H_noisy, V_noisy = add_measurement_noise(H_real, V_real, Del_h=Del_h, Del_V=Del_V)

    # 4. РАСЧЕТ ПРОИЗВОДНОЙ ЗАШУМЛЕННЫХ ДАННЫХ
    print("4. Расчет производной зашумленных данных...")
    dV_noisy = calculate_derivative(H_noisy, V_noisy)

    # 5. СГЛАЖИВАНИЕ
    print("5. Сглаживание данными способами...")

    # Инициализация переменных для сглаженных данных
    V_smooth_scipy = V_noisy.copy()
    V_smooth_orig1 = V_noisy.copy()
    V_smooth_orig2 = V_noisy.copy()

    # 5.1 SciPy сглаживание
    if USE_SCIPY:
        print("   5.1 SciPy сглаживание...")
        try:
            V_smooth_scipy = scipy_smooth(H_noisy, V_noisy, I_beg, I_end,
                                          smooth_factor=scipy_smooth_factor)
            print("      Успешно")
        except Exception as e:
            print(f"      Ошибка: {e}")
    else:
        print("   5.1 SciPy сглаживание: отключено")

    # 5.2 Оригинальный алгоритм Sg_p=1
    if USE_ORIG1:
        print("   5.2 Оригинальный алгоритм Sg_p=1...")
        try:
            smoother1 = OriginalSmoother()  # Отдельный объект для Sg_p=1
            V_smooth_orig1 = smoother1.smooth_data(H_noisy, V_noisy, I_beg, I_end,
                                                   Sg_p=1, **params_orig1)
            print("      Успешно")
        except Exception as e:
            print(f"      Ошибка: {e}")
    else:
        print("   5.2 Оригинальный алгоритм Sg_p=1: отключено")

    # 5.3 Оригинальный алгоритм Sg_p=2
    if USE_ORIG2:
        print("   5.3 Оригинальный алгоритм Sg_p=2...")
        try:
            smoother2 = OriginalSmoother()  # Отдельный объект для Sg_p=2
            V_smooth_orig2 = smoother2.smooth_data(H_noisy, V_noisy, I_beg, I_end,
                                                   Sg_p=2, **params_orig2)
            print("      Успешно")
        except Exception as e:
            print(f"      Ошибка: {e}")
    else:
        print("   5.3 Оригинальный алгоритм Sg_p=2: отключено")

    # 6. РАСЧЕТ ПРОИЗВОДНЫХ СГЛАЖЕННЫХ ДАННЫХ
    print("6. Расчет производных сглаженных данных...")
    dV_scipy = calculate_derivative(H_noisy, V_smooth_scipy)
    dV_orig1 = calculate_derivative(H_noisy, V_smooth_orig1)
    dV_orig2 = calculate_derivative(H_noisy, V_smooth_orig2)

    # 7. ВЫВОД ГРАФИКОВ

    # 7.1 СОЗДАНИЕ ДЕТАЛЬНОЙ СЕТКИ
    def setup_detailed_grid(ax):
        """Настройка детализированной сетки"""
        ax.grid(True, which='major', linestyle='-', alpha=0.7, color='gray')
        ax.grid(True, which='minor', linestyle='--', alpha=0.4, color='lightgray')
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        return ax

    # 7.2 ГРАФИК ОБЪЕМОВ (отдельный)
    print("7.2 Создание и отображение графика объемов...")
    fig_volumes = plt.figure(figsize=(16, 10))
    ax_vol = fig_volumes.add_subplot(111)

    # Данные с кастомизированными стилями
    ax_vol.plot(H_real, V_real, **line_styles['real_data'])
    ax_vol.plot(H_noisy, V_noisy, **line_styles['noisy_data'])

    if USE_SCIPY:
        ax_vol.plot(H_noisy, V_smooth_scipy, **line_styles['scipy_smooth'])
    if USE_ORIG1:
        ax_vol.plot(H_noisy, V_smooth_orig1, **line_styles['orig1_smooth'])
    if USE_ORIG2:
        ax_vol.plot(H_noisy, V_smooth_orig2, **line_styles['orig2_smooth'])

    # Настройки графика
    ax_vol.set_xlabel('Высота H (см)', fontsize=12)
    ax_vol.set_ylabel('Объем V (л)', fontsize=12)
    ax_vol.legend(fontsize=10, loc='lower right')
    ax_vol.set_title('Сравнение алгоритмов сглаживания - Объемы', fontsize=14, fontweight='bold')

    # Детальная сетка для объемов
    ax_vol = setup_detailed_grid(ax_vol)
    ax_vol.yaxis.set_major_locator(MultipleLocator(5000))
    ax_vol.yaxis.set_minor_locator(MultipleLocator(1000))

    plt.tight_layout()

    # Отображение ДО сохранения
    plt.show()
    save_plot(fig_volumes, 'volumes_detailed.png')

    # 7.3 ГРАФИК ПРОИЗВОДНЫХ (отдельный)
    print("7.3 Создание и отображение графика производных...")
    fig_derivatives = plt.figure(figsize=(16, 10))
    ax_der = fig_derivatives.add_subplot(111)

    # Данные с кастомизированными стилями
    ax_der.plot(H_real, dV_real, **line_styles['real_data'])
    ax_der.plot(H_noisy, dV_noisy, **line_styles['noisy_data'])

    if USE_SCIPY:
        ax_der.plot(H_noisy, dV_scipy, **line_styles['scipy_smooth'])
    if USE_ORIG1:
        ax_der.plot(H_noisy, dV_orig1, **line_styles['orig1_smooth'])
    if USE_ORIG2:
        ax_der.plot(H_noisy, dV_orig2, **line_styles['orig2_smooth'])

    # Настройки графика
    ax_der.set_xlabel('Высота H (см)', fontsize=12)
    ax_der.set_ylabel('Производная dV/dH (л/см)', fontsize=12)
    ax_der.legend(fontsize=10, loc='upper right')
    ax_der.set_title('Сравнение алгоритмов сглаживания - Производные', fontsize=14, fontweight='bold')

    # Детальная сетка для производных
    ax_der = setup_detailed_grid(ax_der)

    # Автоматическое определение шага для оси Y производных
    y_range = max(dV_real) - min(dV_real)
    major_step = max(10, y_range / 10)
    minor_step = major_step / 5

    ax_der.yaxis.set_major_locator(MultipleLocator(major_step))
    ax_der.yaxis.set_minor_locator(MultipleLocator(minor_step))

    plt.tight_layout()

    # Отображение ДО сохранения
    plt.show()
    save_plot(fig_derivatives, 'derivatives_detailed.png')

    # 7.4 СОВМЕСТНЫЙ ГРАФИК
    print("7.4 Создание и отображение совместного графика...")
    fig_comparison, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # График V(H) с кастомизированными стилями
    ax1.plot(H_real, V_real, **{**line_styles['real_data'], 'linewidth': 2})
    ax1.plot(H_noisy, V_noisy, **{**line_styles['noisy_data'], 'linewidth': 1.5})
    # ax1.plot(H_noisy, V_smooth_scipy, **{**line_styles['scipy_smooth'], 'linewidth': 1.5})
    # ax1.plot(H_noisy, V_smooth_orig1, **{**line_styles['orig1_smooth'], 'linewidth': 1.5})
    if USE_ORIG2:
        ax1.plot(H_noisy, V_smooth_orig2, **{**line_styles['orig2_smooth'], 'linewidth': 1.5})

    ax1.set_xlabel('Высота H (см)')
    ax1.set_ylabel('Объем V (л)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Сравнение алгоритмов сглаживания - Объемы')

    # График производных с кастомизированными стилями
    ax2.plot(H_real, dV_real, **{**line_styles['real_data'], 'linewidth': 2})
    ax2.plot(H_noisy, dV_noisy, **{**line_styles['noisy_data'], 'linewidth': 1.5})

    if USE_SCIPY:
        ax2.plot(H_noisy, dV_scipy, **{**line_styles['scipy_smooth'], 'linewidth': 1.5})
    if USE_ORIG1:
        ax2.plot(H_noisy, dV_orig1, **{**line_styles['orig1_smooth'], 'linewidth': 1.5})
    if USE_ORIG2:
        ax2.plot(H_noisy, dV_orig2, **{**line_styles['orig2_smooth'], 'linewidth': 1.5})

    ax2.set_xlabel('Высота H (см)')
    ax2.set_ylabel('Производная dV/dH (л/см)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Сравнение алгоритмов сглаживания - Производные')

    plt.tight_layout()

    # Отображение ДО сохранения
    # plt.show()
    save_plot(fig_comparison, 'comparison.png')

    print("✓ Готово! Созданы графики:")
    print("  - plots/pigl/volumes_detailed.png (отдельный график объемов)")
    print("  - plots/pigl/derivatives_detailed.png (отдельный график производных)")
    print("  - plots/pigl/comparison.png (совместный график)")

    print("\nЦВЕТОВАЯ СХЕМА:")
    for key, style in line_styles.items():
        print(f"  {style['label']}: {style['color']}, {style['linestyle']}")


if __name__ == "__main__":
    main()