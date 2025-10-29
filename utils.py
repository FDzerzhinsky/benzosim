# utils.py
"""
Вспомогательные классы
"""

class DataLoaderXLSX():
    """
    Класс для загрузки данных из XLSX файлов.
    Класс принимает путь к файлу, список имён листов и список имён столбцов для загрузки.
    Класс возвращает словарь из датафреймов, где ключами являются имена листов.
    """
    def __init__(self, file_path: str, sheet_names: list, column_names: list):
        """
        Инициализация загрузчика данных из XLSX.

        Args:
            file_path: Путь к XLSX файлу
            sheet_names: Список имён листов для загрузки
            column_names: Список имён столбцов для загрузки
        """
        self.file_path = file_path
        self.sheet_names = sheet_names
        self.column_names = column_names
        self.data = self.load_data()
    def load_data(self) -> dict:
        """
        Загрузка данных из XLSX файла.

        Returns:
            Словарь из датафреймов, где ключами являются имена листов
        """
        import pandas as pd

        data_frames = {}
        for sheet in self.sheet_names:
            df = pd.read_excel(self.file_path, sheet_name=sheet, usecols=self.column_names)
            data_frames[sheet] = df

        return data_frames

