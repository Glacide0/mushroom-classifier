#!/usr/bin/env python
"""Script to directly process mushroom dataset without using Hydra."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Добавляем родительскую директорию в PYTHONPATH
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def main():
    """Process mushroom data directly."""
    # Пути к файлам и директориям
    project_dir = Path(__file__).parent.parent
    raw_data_dir = project_dir / "data" / "raw"
    processed_data_dir = project_dir / "data" / "processed"
    csv_file = raw_data_dir / "mushrooms.csv"
    
    # Проверяем наличие файла
    if not csv_file.exists():
        print(f"Ошибка: CSV файл не найден: {csv_file}")
        return False
    
    print(f"Обработка данных из файла: {csv_file}")
    
    try:
        # Загружаем датасет
        df = pd.read_csv(csv_file)
        print(f"Загружен датасет с {len(df)} строками и {len(df.columns)} столбцами")
        
        # Предполагаем, что первый столбец - целевая переменная
        target_col = df.columns[0]
        
        # Преобразуем категориальную целевую переменную в числовую
        target_mapping = {label: idx for idx, label in enumerate(df[target_col].unique())}
        y = df[target_col].map(target_mapping).values
        
        # Создаем отображение из числовой переменной в имена классов
        class_names = {idx: label for label, idx in target_mapping.items()}
        print(f"Отображение классов: {class_names}")
        
        # Обрабатываем признаки (исключая целевую переменную)
        X_df = df.drop(columns=[target_col])
        
        # One-hot кодирование категориальных признаков
        X_encoded = pd.get_dummies(X_df)
        feature_names = X_encoded.columns.tolist()
        X = X_encoded.values
        
        print(f"Размерность признаков: {X.shape}, Размерность меток: {y.shape}")
        
        # Сохраняем отображение классов
        class_mapping_path = processed_data_dir / "class_mapping.json"
        os.makedirs(processed_data_dir, exist_ok=True)
        
        with open(class_mapping_path, 'w') as f:
            json.dump({str(k): v for k, v in class_names.items()}, f)
            
        print(f"Отображение классов сохранено в {class_mapping_path}")
        
        # Выводим информацию о данных
        print("\nИнформация о датасете:")
        print(f"Количество примеров: {len(df)}")
        print(f"Количество классов: {len(class_names)}")
        print(f"Распределение классов:")
        for class_idx, class_name in class_names.items():
            count = np.sum(y == class_idx)
            percentage = count / len(y) * 100
            print(f"  - {class_name}: {count} ({percentage:.2f}%)")
        
        print("\nДанные успешно обработаны!")
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке данных: {e}")
        return False

if __name__ == "__main__":
    main() 