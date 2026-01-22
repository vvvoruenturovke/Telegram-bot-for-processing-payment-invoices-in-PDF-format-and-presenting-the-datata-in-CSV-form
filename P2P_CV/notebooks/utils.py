
import sys
import os
import matplotlib.pyplot as plt

def setup_notebook():
    """Настройка окружения для ноутбука"""
    sys.path.append('..')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    print("Ноутбук настроен")

def list_test_files(directory="."):
    """Показать тестовые файлы в директории"""
    files = [f for f in os.listdir(directory)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf'))]
    print("Доступные тестовые файлы:")
    for file in files:
        print(f"   - {file}")
    return files