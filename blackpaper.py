import numpy as np
nan = np.nan
a1 = [nan, nan, 1, 2, 3, nan]
a2 = [3, 4, 2, nan, nan, nan]
a3 = [nan, 5, 3, 4, 3, 2]
res = [3, 4.5, 2, 3, 3, 2]

# Собираем ВСЕ массивы
arrays = [a1, a2, a3]
result = []

# Для каждой позиции (высоты)
for i in range(len(arrays[0])):
    values_at_height_i = []

    # Собираем значения из всех массивов на этой высоте
    for arr in arrays:
        if not np.isnan(arr[i]):  # Если значение существует
            values_at_height_i.append(arr[i])

    if values_at_height_i:  # Если есть хоть одно значение
        result.append(np.mean(values_at_height_i))
    else:
        result.append(np.nan)  # Или пропуск
print(result)