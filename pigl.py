from utils import DataLoaderXLSX
import pandas as pd
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
loader = DataLoaderXLSX('data/53063_export.xlsx', ['storage_level'], ['DATE_TIME', 'STOR_ID', 'OIL_LEVEL', 'VOLUME', 'DELTA'])
data = loader.data


rawdata = data['storage_level']
storages = list(set(rawdata['STOR_ID'].tolist()))
data_storages = dict()
for stor_id in storages:
    data_storages[stor_id] = rawdata[rawdata['STOR_ID'] == stor_id]

x = data_storages[storages[0]]['OIL_LEVEL']
y = data_storages[storages[0]]['DELTA']

plt.figure(figsize=(15, 8))  # Set the canvas size to match the one in `visualization.py`

plt.plot(x, y, 'b-', linewidth=2, label='Градуировка')  # Plot x (OIL_LEVEL) against y (VOLUME)

plt.xlabel('Высота H, см')  # Label for the x-axis
plt.ylabel('Коэфф. вместимости, л/см')  # Label for the y-axis
plt.title('Градуировочная характеристика')  # Title of the plot
plt.legend()  # Add a legend
plt.grid(True, alpha=0.3)  # Add a grid with transparency

plt.show()  # Display the plot