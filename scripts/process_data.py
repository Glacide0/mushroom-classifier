#!/usr/bin/env python
"""Script to process data."""

from pathlib import Path
import sys
import os

# Добавляем родительскую директорию в PYTHONPATH для импорта модулей
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from omegaconf import OmegaConf
    from mushroom_classifier.data import MushroomDataModule
    
    def main():
        """Process data using MushroomDataModule."""
        # Загружаем конфигурацию
        config_path = Path(parent_dir) / "configs" / "config.yaml"
        if not config_path.exists():
            print(f"Ошибка: Файл конфигурации не найден: {config_path}")
            return
        
        config = OmegaConf.load(config_path)
        print("Конфигурация загружена успешно.")
        
        # Создаем экземпляр MushroomDataModule
        data_module = MushroomDataModule(config)
        print("Запуск обработки данных...")
        
        # Обрабатываем данные
        data_module.process_data()
        print("Обработка данных завершена.")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все необходимые зависимости установлены.")
except Exception as e:
    print(f"Произошла ошибка: {e}") 