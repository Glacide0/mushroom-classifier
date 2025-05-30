#!/usr/bin/env python
"""Script to set up data directories and initialize DVC."""

import os
import subprocess
import sys
from pathlib import Path
import shutil
import requests

def download_from_gdrive(file_id, output_path):
    """Загрузка файла с Google Drive.
    
    Args:
        file_id: ID файла на Google Drive
        output_path: Путь для сохранения файла
    
    Returns:
        bool: True если загрузка успешна, False в противном случае
    """
    # Прямая ссылка на датасет грибов
    direct_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
    
    print(f"Попытка загрузки данных напрямую с UCI Machine Learning Repository...")
    try:
        response = requests.get(direct_url)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"Файл успешно загружен и сохранен в {output_path}")
            return True
        else:
            print(f"Ошибка при загрузке: статус {response.status_code}")
    except Exception as e:
        print(f"Ошибка при загрузке напрямую: {e}")
        
    # Попытка загрузки с Google Drive через gdown
    try:
        try:
            import gdown
            print(f"Загрузка файла с Google Drive (ID: {file_id})...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
            if os.path.exists(output_path):
                return True
        except ImportError:
            # Если gdown не установлен, пробуем установить
            print("Установка gdown...")
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
            import gdown
            print(f"Загрузка файла с Google Drive (ID: {file_id})...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
            if os.path.exists(output_path):
                return True
    except Exception as e:
        print(f"Ошибка при загрузке с Google Drive: {e}")
    
    # Попытка использовать альтернативный метод
    try:
        print("Попытка загрузки через curl...")
        subprocess.run([
            "curl", "-L",
            f"https://drive.google.com/uc?export=download&id={file_id}",
            "-o", output_path
        ], check=True)
        if os.path.exists(output_path):
            return True
    except Exception as curl_e:
        print(f"Ошибка при загрузке с помощью curl: {curl_e}")
    
    # Если все методы не сработали, предлагаем перейти по ссылке вручную
    print("\nВы можете загрузить файл вручную по одной из следующих ссылок:")
    print(f"1. UCI Repository: {direct_url}")
    print(f"2. Google Drive: https://drive.google.com/file/d/{file_id}/view")
    print("После загрузки сохраните файл как 'mushrooms.csv' в директорию data/raw")
    
    return False

def main():
    """Set up data directories and initialize DVC."""
    # Путь к директории проекта (перейти на уровень выше от scripts)
    project_dir = Path(__file__).parent.parent

    # Пути к директориям данных
    data_dir = project_dir / "data"
    raw_data_dir = data_dir / "raw"
    processed_data_dir = data_dir / "processed"
    train_dir = processed_data_dir / "train"
    val_dir = processed_data_dir / "val"
    test_dir = processed_data_dir / "test"
    
    # Создать директории для данных
    print(f"Создание директорий для данных...")
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Создать директории для моделей и логов
    models_dir = project_dir / "models"
    checkpoints_dir = models_dir / "checkpoints"
    exported_dir = models_dir / "exported"
    logs_dir = project_dir / "logs" 
    mlflow_dir = logs_dir / "mlflow"
    plots_dir = project_dir / "plots"
    
    print(f"Создание директорий для моделей и логов...")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(exported_dir, exist_ok=True)
    os.makedirs(mlflow_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Целевой путь для CSV файла
    target_csv = raw_data_dir / "mushrooms.csv"
    
    # 1. Сначала проверим наличие CSV файла
    if not os.path.exists(target_csv):
        # Попытка загрузки с Google Drive
        gdrive_file_id = "1WVMvdjL6Q_vUD_Lb-jbEfPaDrT7pEerf"  # ID файла на Google Drive
        
        if not download_from_gdrive(gdrive_file_id, str(target_csv)):
            # 2. Если загрузка с источников не удалась, ищем локальные пути
            print("Загрузка с онлайн-источников не удалась. Проверка локальных путей...")
            
            # Пути для локальной работы на Windows
            possible_paths = [
                "../dataset/mushrooms.csv",                 # Относительный путь
                "D:/coding projects/dataset/mushrooms.csv", # Абсолютный путь
                str(Path(project_dir).parent / "dataset" / "mushrooms.csv"), # Родительская директория
            ]
            
            csv_found = False
            for csv_path in possible_paths:
                if os.path.exists(csv_path):
                    csv_file = csv_path
                    csv_found = True
                    print(f"CSV файл найден: {csv_file}")
                    
                    try:
                        shutil.copy(csv_file, target_csv)
                        print(f"CSV файл скопирован в {target_csv}")
                        break
                    except Exception as e:
                        print(f"Ошибка при копировании CSV файла: {e}")
            
            # 3. Если ни один из методов не сработал, запрашиваем путь вручную
            if not csv_found and not os.path.exists(target_csv):
                print("ВНИМАНИЕ: CSV файл не найден автоматически.")
                print("\nПожалуйста, укажите путь к CSV файлу (или оставьте пустым, чтобы пропустить):")
                custom_path = input("Путь к CSV файлу: ").strip()
                
                if custom_path:
                    if os.path.exists(custom_path):
                        try:
                            shutil.copy(custom_path, target_csv)
                            print(f"CSV файл скопирован в {target_csv}")
                        except Exception as e:
                            print(f"Ошибка при копировании CSV файла: {e}")
                    else:
                        print(f"Файл не найден: {custom_path}")
                        print("Продолжаем без данных...")
    else:
        print(f"CSV файл уже существует по пути: {target_csv}")
        
    # Проверка и инициализация Git и DVC если нужно
    try:
        git_dir = project_dir / ".git"
        if not os.path.exists(git_dir):
            print("Git репозиторий не найден. Инициализируем Git...")
            subprocess.run(["git", "init"], cwd=project_dir, check=True)
            
            # Создаем .gitignore
            gitignore_path = project_dir / ".gitignore"
            if not os.path.exists(gitignore_path):
                with open(gitignore_path, "w") as f:
                    f.write("/data\n/models\n/logs\n/__pycache__/\n*.pyc\n")
                print("Создан .gitignore файл")
            
            # Делаем первый коммит
            subprocess.run(["git", "add", ".gitignore"], cwd=project_dir, check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=project_dir, check=True)
            print("Git инициализирован и создан первый коммит")
        else:
            print("Git репозиторий уже инициализирован.")
        
        # Инициализация DVC
        dvc_dir = project_dir / ".dvc"
        if not os.path.exists(dvc_dir):
            print("Инициализация DVC...")
            subprocess.run(["dvc", "init"], cwd=project_dir, check=True)
        else:
            print("DVC уже инициализирован.")
            
        # Добавить данные в DVC
        if os.path.exists(raw_data_dir / "mushrooms.csv"):
            print(f"Добавление данных в DVC...")
            try:
                subprocess.run(["dvc", "add", str(raw_data_dir)], cwd=project_dir, check=True)
                print("Данные успешно добавлены в DVC.")
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при добавлении данных в DVC: {e}")
                print("Продолжаем работу...")
        else:
            print("Данные не найдены, пропускаем добавление в DVC.")
            
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при инициализации Git или DVC: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
    
    print("\nНастройка данных завершена!")
    print("Теперь вы можете запустить подготовку данных.")

if __name__ == "__main__":
    main() 