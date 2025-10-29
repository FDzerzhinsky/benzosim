import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
def bspline_(cloud):
    print('Начинаем построение сплайна...\n')
    n = cloud.shape[0] -1 # 1028 Размер массива исходных данных 0..n
    # n = 1028

    # Число узлов в базовой сетке -3,-2,...,Ns+1,Ns+2,Ns+3

    Ns = 6 + math.ceil(cloud.max() - cloud.min())
    #FIXME: Убрать из релиза, устарело
    # Ns = 6 + math.ceil(cloud['Уровень, см'].max() - cloud['Уровень, см'].min())

    # X0 = 102.0

    H = 1.0  # Шаг базовой сетки

    Y = np.zeros((n + 1, Ns + 3))  # Массив скалярных произведений
    m = np.zeros(Ns + 1, dtype=int)  # Номера узлов границ интервалов базовой сетки
    Xs = np.zeros(Ns + 3)  # Координаты узлов базовой сетки
    A = np.zeros((Ns + 2, Ns + 2))  # Матрица значений скалярных произведений
    A_ = np.zeros((Ns + 3, Ns + 3))  # Матрица для решения СЛАУ
    R_p = np.zeros(Ns + 2)  # Вектор скалярных произведений для правой части СЛАУ
    b = np.zeros(Ns + 2)  # Коэффициенты разложения по сплайн-базису
    x = np.zeros(n + 1)  # Массивы исходных данных
    f = np.zeros(n + 1)  # Массивы исходных данных
    d = np.zeros(Ns + 3)  # Вектор правой части для решения СЛАУ
    z = np.zeros(Ns + 3)  # Вектор правой части для решения СЛАУ
    i, j, k = 0, 0, 0

    def Bs(t: float, Num: int) -> float:  # Значение базисного сплайна
        if Num == 1:
            return t * t * t / 6
        elif Num == 2:
            return 1.0 / 6 + 1.0 / 2 * t * (1 + t * (1 - t))
        elif Num == 3:
            return 1.0 / 6 + 1.0 / 2 * (1 - t) * (1 + t * (1 - t))
        elif Num == 4:
            return (1 - t) * (1 - t) * (1 - t) / 6
        else: return None
    def C_Y(N: int, k: int, x: np.ndarray):  # Построение векторов значений базисных функций в узлах исходной сетки
        for i in range(N + 1):
            Y[i, k] = 0
            for Lk in range(-2, 2):
                XL = X0 + (k + Lk) * H
                XL1 = X0 + (k + Lk + 1) * H
                if XL <= x[i] < XL1:
                    Y[i, k] = Bs((x[i] - XL) / H, Lk + 3)

    def MultMatr(n: int, A: np.ndarray, B: np.ndarray, C: np.ndarray):  # Умножение матрицы
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                S = 0
                for k in range(1, n + 1):
                    S += A[i, k] * B[k, j]
                C[i, j] = S

    def LU(n: int, A: np.ndarray, b: np.ndarray,
           x: np.ndarray):  # Решение СЛАУ методом LU-разложения
        Eps = 1e-4
        U = A.copy()
        P = np.arange(n + 3)
        L = np.zeros((n + 1, n + 1))
        Det = 1
        k = 0
        while True:
            k += 1
            m = k
            MaxA = abs(U[k, k])
            for i in range(k + 1, n + 1):
                if abs(U[i, k]) > MaxA:
                    m = i
                    MaxA = abs(U[i, k])

            if MaxA >= Eps:
                if m != k:  # Перестановка строк
                    Det = -Det
                    P[k], P[m] = P[m], P[k]
                    U[[m, k], :] = U[[k, m], :]
                    A[[m, k], :] = A[[k, m], :]

                for j in range(k + 1, n + 1):
                    U[k, j] /= U[k, k]
                U[k, k] = 1

                for i in range(k + 1, n + 1):
                    for j in range(n, k - 1, -1):
                        U[i, j] -= U[k, j] * U[i, k]

                for j in range(1, k + 1):
                    L[k, j] = A[k, j]
                    for i in range(1, j):
                        L[k, j] -= L[k, i] * U[i, j]
            else:
                break

        for i in range(1, n + 1):
            Det *= L[i, i]

        y = b.copy()
        for i in range(1, n + 1):
            b[i] = y[P[i]]

        for i in range(1, n + 1):
            S = 0
            for j in range(1, i):
                S += L[i, j] * y[j]
            y[i] = (b[i] - S) / L[i, i]

        for i in range(n, 0, -1):
            S = 0
            for j in range(i + 1, n + 1):
                S += U[i, j] * x[j]
            x[i] = y[i] - S

    # Ввод исходных данных
    # with open('Inp_S.txt', 'r') as Inp:
    #     print('---- Исходные данные -------')
    #     print('    i  x[i]    f[i]')
    #     for i in range(n + 1):
    #         x[i], f[i] = map(float, Inp.readline().split())
    #         print(f'{i:5} {x[i]:4.2f}  {f[i]:8.2f}')

    x, f = np.flip(cloud.index.to_numpy()), np.flip(cloud.values)  # Ввод исходных данных

    #FIXME: Вырезать это легаси из релиза
    # x, f = np.flip(cloud['Уровень, см'].to_numpy()), np.flip(cloud['Накопительная'].to_numpy()) # Ввод исходных данных
    # x, f = cloud['Уровень, см'].to_numpy(), cloud['Накопительная'].to_numpy()  # Ввод исходных данных
    X0 = np.floor(min(x))
    # Базовая сетка
    # print()
    for k in range(-2, Ns + 3):
        Xs[k] = X0 + k * H
    # Попадание в интервал базовой сетки
    m[0] = 0
    k = 1
    #for i in range(n + 1):
    for i in range(n):
        if x[i] > Xs[k]:
            m[k] = i
            k += 1
    m[k] = n + 1
    Nk = k
    # print()
    # print('--- Число точек в интервале базовой сетки -----------')
    # print()
    # for k in range(1, Nk + 1):
    #     print(f' [{Xs[k - 1]:5.2f} {Xs[k]:5.2f}] : {m[k] - m[k - 1]:3}')
    # print('----- Попадание в интервалы базовой сетки-------')
    # for k in range(Nk):
    #     print(f' [{k:3}]   {{{Xs[k]:5.2f} {Xs[k + 1]:5.2f}}} : ', end='')
    #     for i in range(m[k], m[k + 1]):
    #         print(f'{x[i]:5.2f}  ', end='')
    #     print()
    # print(f' Nk={Nk:2}')
    for k in range(-1, Nk + 2):
        C_Y(n, k, x)
    # print('-------Матрица скалярных произведений-----------------')
    for k in range(Nk + 1):
        for j in range(-1, Nk + 2):
            A[k, j] = 0
            for i in range(n + 1):
                A[k, j] += Y[i, k] * Y[i, j]
        #     print(f' {A[k, j]:7.4f}', end=' ')
        # print()
    # Доп. условия
    for j in range(2, Ns - 1):
        A[-1, j] = 0
        A[Nk + 1, j] = 0
    A[-1, -1] = 1
    A[-1, 0] = -2
    A[-1, 1] = 1
    A[Nk + 1, Nk - 1] = 1
    A[Nk + 1, Nk] = -2
    A[Nk + 1, Nk + 1] = 1
    # print()
    # print(' Правая часть')
    for k in range(-1, Nk + 2):
        R_p[k] = 0
        for i in range(n + 1):
            R_p[k] += Y[i, k] * f[i]
        # print(f' {R_p[k]:7.3f}', end=' ')
    R_p[-1] = 0
    R_p[Nk + 1] = 0
    # print()
    # Подготовка к решению СЛАУ
    for k in range(-1, Nk + 2):
        for j in range(-1, Nk + 2):
            A_[k + 2, j + 2] = A[k, j]
        d[k + 2] = R_p[k]
    # print(' Данные для решения СЛАУ')
    # for k in range(1, Nk + 4):
    #     for j in range(1, Nk + 4):
    #         print(f' {A_[k, j]:7.3f}', end=' ')
    #     print(f' | {d[k]:7.1f}')
    LU(Nk + 3, A_, d, z)
    print('---- Решение ----')
    # for k in range(1, Nk + 4):
    #     print(f' {z[k]:7.1f}', end=' ')
    # print()
    for k in range(-1, Nk + 2):
        b[k] = z[k + 2]
    # print('------ Интерполяция---------')
    # print('   Xs     S(Xs)     dS(Xs)/dx ')
    H_, chart, dChart = [], [], [] # переменные-списки возврата сплайна и его производной
    for k in range(Nk + 1):
        H_.append(X0 + k * H)
        chart.append((b[k - 1] + 4 * b[k] + b[k + 1]) / 6)
        dChart.append((b[k + 1] - b[k - 1]) / (2 * H))


    #     print(
    #         f' {X0 + k * H:5.2f}  {(b[k - 1] + 4 * b[k] + b[k + 1]) / 6:7.1f}  {(b[k + 1] - b[k - 1]) / (2 * H):7.1f}')
    #
    # print()
    # with open('Rez-main_Python.txt', 'w') as Out:
    #     for k in range(Nk + 1):
    #         print(
    #             f'  {X0 + k * H:5.2f}  {(b[k - 1] + 4 * b[k] + b[k + 1]) / 6:7.1f}  {(b[k + 1] - b[k - 1]) / (2 * H):7.1f}',
    #             file=Out)
    print('Сплайн построен.\n')
    H_.pop(-1), chart.pop(-1), dChart.pop(-1)  # Удаляем последний элемент, т.к. он не нужен
    return np.array(H_), np.array(chart), np.array(dChart)



# H_, chart, dChart = bspline_()
# plt.plot(np.array(H_), np.array(dChart))
# plt.show()




