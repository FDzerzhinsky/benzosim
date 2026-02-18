#!/usr/bin/env python3
import csv
import math

# Читаем результаты
f = open('plots/periodic_Тестовый_эксперимент_(конфиг_из_config.py)/combined_scipy.csv')
lines = list(csv.reader(f))
f.close()

print(f"Total lines (including header): {len(lines)}")
print(f"First data row: H={lines[1][0]} cm, V={lines[1][1]} л")
print(f"Last data row: H={lines[-1][0]} cm, V={lines[-1][1]} л")

# Вычисляем идеальные значения для сравнения
R = 100
L = 500

def V_h(h_norm):
    if h_norm <= 1.0:
        alfa = 2 * math.acos(1.0 - h_norm - 1.0e-14)
        V = 0.5 * (alfa - math.sin(alfa))
    else:
        alfa = 2 * math.acos(h_norm - 1.0 - 1.0e-14)
        V = 0.5 * (2 * math.pi - alfa + math.sin(alfa))
    return V

# Проверяем несколько точек
print("\nСравнение восстановленной V с идеальной:")
print("H_см  | V_восст | V_идеал | ошибка%")
print("-" * 45)
for i in [1, 10, 20, 30, 40, -1]:
    if i < len(lines):
        H = float(lines[i][0])
        V_rec = float(lines[i][1])
        V_ideal = R * R * L * V_h(H / R) / 1000
        error_pct = abs(V_rec - V_ideal) / V_ideal * 100
        print(f"{H:6.1f} | {V_rec:7.1f} | {V_ideal:7.1f} | {error_pct:6.1f}%")

# Проверим диапазон
print(f"\n\n📊 ДИАПАЗОН РЕЗУЛЬТАТОВ:")
print(f"  Минимум H: {float(lines[1][0]):.1f} см")
print(f"  Максимум H: {float(lines[-1][0]):.1f} см")
print(f"  Диапазон: {float(lines[-1][0]) - float(lines[1][0]):.1f} см (из 200 см)")

# Проверим облако точек
f = open('plots/periodic_Тестовый_эксперимент_(конфиг_из_config.py)/counts_scipy.csv')
counts_lines = list(csv.reader(f))
f.close()

# Посчитаем распределение counts
counts_dict = {}
for i in range(1, len(counts_lines)):
    h = float(counts_lines[i][0])
    count = int(counts_lines[i][1])
    if count not in counts_dict:
        counts_dict[count] = 0
    counts_dict[count] += 1

print(f"\n📊 РАСПРЕДЕЛЕНИЕ ОБЛАКА ТОЧЕК:")
print(f"  Количество периодов, прошедших через узел:")
for count in sorted(counts_dict.keys()):
    pct = counts_dict[count] / len(counts_lines) * 100
    print(f"    count={count}: {counts_dict[count]} узлов ({pct:.1f}%)")

