"""
Вспомогательные функции для приложения.
"""
import os
import sys
import shutil
import subprocess
import json
import csv
import datetime
import hashlib
import pytesseract
from PyQt6.QtWidgets import QMessageBox, QFileDialog
from PyQt6.QtCore import QStandardPaths
from PIL import Image

from . import config


def show_error_message(parent, title, message):
    """Показать сообщение об ошибке."""
    QMessageBox.critical(parent, title, message)


def show_info_message(parent, title, message):
    """Показать информационное сообщение."""
    QMessageBox.information(parent, title, message)


def show_warning_message(parent, title, message):
    """Показать предупреждающее сообщение."""
    QMessageBox.warning(parent, title, message)


def show_question_message(parent, title, message):
    """Показать вопросительное сообщение с вариантами Да/Нет."""
    return QMessageBox.question(parent, title, message, 
                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes


def get_open_file_path(parent, caption, directory, filter_):
    """Показать диалог выбора файла и получить путь."""
    options = QFileDialog.Option.ReadOnly
    file_path, _ = QFileDialog.getOpenFileName(
        parent, caption, directory, filter_, options=options
    )
    return file_path


def get_save_file_path(parent, caption, directory, filter_):
    """Показать диалог сохранения файла и получить путь."""
    options = QFileDialog.Option.ReadOnly
    file_path, _ = QFileDialog.getSaveFileName(
        parent, caption, directory, filter_, options=options
    )
    return file_path


def get_documents_dir():
    """Получить директорию для документов пользователя."""
    return QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)


def is_tesseract_installed():
    """
    Проверяет наличие Tesseract в системе.
    
    Returns:
        bool: True, если Tesseract найден, иначе False
    """
    try:
        # Если задан путь, используем его
        if config.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_PATH
            pytesseract.get_tesseract_version()
            return True
        
        # Пытаемся найти tesseract в PATH
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def find_tesseract_in_path():
    """
    Пытается найти путь к исполняемому файлу Tesseract.
    
    Returns:
        str: Путь к tesseract.exe, если найден, иначе None
    """
    # Проверяем известные пути установки (относительные пути убраны для безопасности)
    potential_paths = [
        # Поиск в PATH (приоритет)
        shutil.which("tesseract"),
        # Windows пути (стандартные установки)
        "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        # Linux/macOS стандартные пути
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",  # macOS ARM Homebrew
        "/usr/local/Cellar/tesseract/*/bin/tesseract"  # macOS Intel Homebrew
    ]
    
    for path in potential_paths:
        if path and os.path.isfile(path):
            try:
                # Проверяем, что это действительно Tesseract
                old_cmd = pytesseract.pytesseract.tesseract_cmd
                pytesseract.pytesseract.tesseract_cmd = path
                pytesseract.get_tesseract_version()
                return path
            except:
                # Если это не Tesseract, восстанавливаем путь и продолжаем поиск
                pytesseract.pytesseract.tesseract_cmd = old_cmd
    
    return None


def convert_pdf_to_image(pdf_path, output_folder=None):
    """
    Конвертирует первую страницу PDF-файла в изображение.
    
    Args:
        pdf_path (str): Путь к PDF-файлу
        output_folder (str, optional): Папка для сохранения изображения. 
                                     По умолчанию используется временная папка.
                                     
    Returns:
        str: Путь к созданному изображению или None в случае ошибки
    """
    try:
        # Для отложенного импорта, чтобы не требовать pdf2image при старте приложения
        from pdf2image import convert_from_path
        
        # Если папка не указана, используем временную
        if not output_folder:
            output_folder = config.TEMP_PATH
            os.makedirs(output_folder, exist_ok=True)
        
        # Конвертируем только первую страницу
        images = convert_from_path(pdf_path, first_page=1, last_page=1)
        
        if images:
            # Создаем имя выходного файла на основе входного
            basename = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = os.path.join(output_folder, f"{basename}_page1.jpg")
            
            # Сохраняем изображение
            images[0].save(output_path, "JPEG")
            return output_path
        
        return None
    except Exception as e:
        print(f"Ошибка при конвертации PDF в изображение: {str(e)}")
        return None


def get_extension(file_path):
    """Получить расширение файла в нижнем регистре."""
    return os.path.splitext(file_path)[1].lower()


def is_supported_format(file_path):
    """Проверить, поддерживается ли формат файла."""
    ext = get_extension(file_path)
    return ext in config.SUPPORTED_FORMATS


def is_image_format(file_path):
    """Проверить, является ли файл изображением."""
    ext = get_extension(file_path)
    return ext in config.SUPPORTED_IMAGE_FORMATS


def is_pdf_format(file_path):
    """Проверить, является ли файл PDF."""
    ext = get_extension(file_path)
    return ext in config.SUPPORTED_PDF_FORMAT


def is_valid_file(file_path):
    """
    Проверяет, является ли файл поддерживаемым форматом (изображение или PDF).
    
    Args:
        file_path (str): Путь к файлу
        
    Returns:
        bool: True, если формат файла поддерживается, иначе False
    """
    if not os.path.isfile(file_path):
        return False
    
    ext = os.path.splitext(file_path)[1].lower()
    return ext in config.SUPPORTED_FORMATS


def _export_single_to_json(data, output_path):
    """Экспортирует один словарь в JSON."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True, "Экспорт в JSON успешен"
    except Exception as e:
        msg = f"Ошибка при экспорте в JSON: {str(e)}"
        print(msg)
        return False, msg


def _export_single_to_html(data, output_path):
    """Экспортирует один словарь в HTML (простой формат)."""
    try:
        html_content = f"""<!DOCTYPE html>
<html lang=\"ru\">
<head>
    <meta charset=\"UTF-8\">
    <title>Результат обработки счета</title>
    <style>
        body {{ font-family: sans-serif; }}
        dl {{ border: 1px solid #ccc; padding: 10px; }}
        dt {{ font-weight: bold; float: left; width: 150px; clear: left; }}
        dd {{ margin-left: 160px; margin-bottom: 5px; }}
    </style>
</head>
<body>
    <h1>Результат обработки счета</h1>
    <dl>
"""
        for key, value in data.items():
            # Преобразуем переносы строк в <br> для HTML
            display_value = str(value).replace('\n', '<br>')
            html_content += f'        <dt>{key}:</dt><dd>{display_value}</dd>\n'
        
        html_content += """    </dl>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return True, "Экспорт в HTML успешен"
    except Exception as e:
        msg = f"Ошибка при экспорте в HTML: {str(e)}"
        print(msg)
        return False, msg


def _export_single_to_csv(data, output_path):
    """Экспортирует один словарь в CSV."""
    try:
        headers = list(data.keys())
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f: # utf-8-sig для Excel
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=';') # Используем точку с запятой
            writer.writeheader()
            writer.writerow(data)
        return True, "Экспорт в CSV успешен"
    except Exception as e:
        msg = f"Ошибка при экспорте в CSV: {str(e)}"
        print(msg)
        return False, msg


def _export_single_to_txt(data, output_path):
    """Экспортирует один словарь в TXT."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
        return True, "Экспорт в TXT успешен"
    except Exception as e:
        msg = f"Ошибка при экспорте в TXT: {str(e)}"
        print(msg)
        return False, msg


def _export_batch_to_json(data_list, output_path):
    """Экспортирует список словарей в JSON."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
        return True, "Пакетный экспорт в JSON успешен"
    except Exception as e:
        msg = f"Ошибка при пакетном экспорте в JSON: {str(e)}"
        print(msg)
        return False, msg


def _export_batch_to_html(data_list, output_path):
    """Экспортирует список словарей в HTML таблицу."""
    if not data_list: 
        return False, "Нет данных для экспорта в HTML."
    try:
        headers = list(data_list[0].keys())
        html_content = f"""<!DOCTYPE html>
<html lang=\"ru\">
<head>
    <meta charset=\"UTF-8\">
    <title>Результаты пакетной обработки</title>
    <style>
        body {{ font-family: sans-serif; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 0.9em; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: top; }}
        th {{ background-color: #f2f2f2; position: sticky; top: 0; z-index: 1; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h1>Результаты пакетной обработки</h1>
    <table>
        <thead>
            <tr>
"""
        for header in headers:
             html_content += f'                <th>{header}</th>\n'
        html_content += """            </tr>
        </thead>
        <tbody>
"""
        for row_dict in data_list:
            html_content += '            <tr>\n'
            for header in headers:
                value = row_dict.get(header, '')
                # Преобразуем переносы строк в <br> для HTML
                display_value = str(value).replace('\n', '<br>')
                html_content += f'                <td>{display_value}</td>\n'
            html_content += '            </tr>\n'

        html_content += """        </tbody>
    </table>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return True, "Пакетный экспорт в HTML успешен"
    except Exception as e:
        msg = f"Ошибка при пакетном экспорте в HTML: {str(e)}"
        print(msg)
        return False, msg


def _export_batch_to_csv(data_list, output_path):
    """Экспортирует список словарей в CSV."""
    if not data_list: 
        return False, "Нет данных для экспорта в CSV."
    try:
        headers = list(data_list[0].keys()) # Получаем ключи из первого словаря как заголовки
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=';')
            writer.writeheader()
            writer.writerows(data_list) # Записываем все строки
        return True, "Пакетный экспорт в CSV успешен"
    except Exception as e:
        msg = f"Ошибка при пакетном экспорте в CSV: {str(e)}"
        print(msg)
        return False, msg


def _export_batch_to_txt(data_list, output_path):
    """Экспортирует список словарей в TXT."""
    if not data_list: 
        return False, "Нет данных для экспорта в TXT."
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, data_dict in enumerate(data_list):
                f.write(f"--- Запись {i+1} ---\n")
                for key, value in data_dict.items():
                    f.write(f"{key}: {value}\n")
                f.write("--------------------\n\n")
        return True, "Пакетный экспорт в TXT успешен"
    except Exception as e:
        msg = f"Ошибка при пакетном экспорте в TXT: {str(e)}"
        print(msg)
        return False, msg


def export_results(results_data, output_path, format_type):
    """
    Экспортирует результаты обработки в указанный формат.
    Поддерживает экспорт как одиночного результата (dict), так и пакетного (list of dicts).

    Args:
        results_data (dict or list): Данные для экспорта.
        output_path (str): Путь для сохранения файла.
        format_type (str): Формат экспорта ('txt', 'json', 'html', 'csv').

    Returns:
        tuple: (bool, str) - Успех (True/False) и сообщение.
    """
    if not results_data:
        return False, "Нет данных для экспорта."

    is_batch = isinstance(results_data, list)

    print(f"Экспорт данных (Пакетный: {is_batch}) в формат: {format_type}, путь: {output_path}")

    try:
        if format_type == 'json':
            if is_batch:
                return _export_batch_to_json(results_data, output_path)
            else:
                return _export_single_to_json(results_data, output_path)
        elif format_type == 'html':
            if is_batch:
                return _export_batch_to_html(results_data, output_path)
            else:
                return _export_single_to_html(results_data, output_path)
        elif format_type == 'csv':
            if is_batch:
                return _export_batch_to_csv(results_data, output_path)
            else:
                 # Убедимся, что для одиночного CSV данные обернуты в список
                 if isinstance(results_data, dict):
                     return _export_batch_to_csv([results_data], output_path) 
                 else:
                     return False, "Некорректный формат данных для CSV экспорта."
        elif format_type == 'txt':
            if is_batch:
                return _export_batch_to_txt(results_data, output_path)
            else:
                return _export_single_to_txt(results_data, output_path)
        else:
            return False, f"Неподдерживаемый формат экспорта: {format_type}"
    except Exception as e:
        # Общая ошибка на случай непредвиденных ситуаций
        msg = f"Непредвиденная ошибка при экспорте в {format_type}: {str(e)}"
        print(msg)
        import traceback
        traceback.print_exc()
        return False, msg


def check_file_integrity(file_path):
    """
    Проверяет целостность файла, вычисляя его хеш.
    
    Args:
        file_path (str): Путь к проверяемому файлу
        
    Returns:
        str: MD5-хеш файла или None, если файл не существует
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Ошибка при проверке целостности файла {file_path}: {str(e)}")
        return None


def check_project_integrity():
    """
    Проверяет целостность всех файлов проекта.
    
    Returns:
        dict: Словарь с путями к файлам и их хешами
    """
    project_files = []
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Собираем все .py файлы
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                project_files.append(os.path.join(root, file))
    
    # Проверяем целостность
    integrity_data = {}
    for file_path in project_files:
        rel_path = os.path.relpath(file_path, project_dir)
        integrity_data[rel_path] = check_file_integrity(file_path)
    
    return integrity_data


def get_model_cache_info():
    """
    Получает информацию о кэшированных моделях.
    
    Returns:
        dict: Словарь с информацией о моделях в кэше
    """
    models_info = {}
    
    # Создаем папку для моделей, если она не существует
    os.makedirs(config.MODELS_PATH, exist_ok=True)
    
    # Проверяем наличие моделей LayoutLM
    layoutlm_dir = os.path.join(config.MODELS_PATH, 'layoutlm')
    if os.path.exists(layoutlm_dir):
        models_info['layoutlm'] = {
            'path': layoutlm_dir,
            'size': get_dir_size(layoutlm_dir),
            'last_modified': get_last_modified_time(layoutlm_dir),
            'is_complete': is_model_complete(layoutlm_dir, 'layoutlm')
        }
    
    # Проверяем наличие моделей Donut
    donut_dir = os.path.join(config.MODELS_PATH, 'donut')
    if os.path.exists(donut_dir):
        models_info['donut'] = {
            'path': donut_dir,
            'size': get_dir_size(donut_dir),
            'last_modified': get_last_modified_time(donut_dir),
            'is_complete': is_model_complete(donut_dir, 'donut')
        }
    
    return models_info


def get_dir_size(path):
    """
    Получает размер директории в байтах.
    
    Args:
        path (str): Путь к директории
        
    Returns:
        int: Размер директории в байтах
    """
    total_size = 0
    if not os.path.exists(path):
        return 0
    
    if os.path.isfile(path):
        return os.path.getsize(path)
    
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    # Игнорируем символические ссылки
                    if not os.path.islink(file_path):
                        total_size += os.path.getsize(file_path)
                except (FileNotFoundError, PermissionError):
                    continue
    except (PermissionError, OSError) as e:
        print(f"Ошибка при получении размера директории {path}: {str(e)}")
    
    return total_size


def get_last_modified_time(path):
    """
    Получает время последнего изменения файла или директории.
    
    Args:
        path (str): Путь к файлу или директории
        
    Returns:
        float: Время последнего изменения в секундах с начала эпохи
    """
    if not os.path.exists(path):
        return 0
    
    last_modified = os.path.getmtime(path)
    
    # Если это директория, проверяем также файлы внутри
    if os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    mtime = os.path.getmtime(fp)
                    if mtime > last_modified:
                        last_modified = mtime
    
    return last_modified


def is_model_complete(model_dir, model_type):
    """
    Проверяет, что модель полностью загружена.
    
    Args:
        model_dir (str): Путь к директории модели
        model_type (str): Тип модели ('layoutlm' или 'donut')
        
    Returns:
        bool: True, если модель полностью загружена, иначе False
    """
    # Проверяем наличие необходимых файлов в зависимости от типа модели
    if model_type.lower() == 'layoutlm':
        required_files = ['config.json', 'pytorch_model.bin']
    else:  # donut
        required_files = ['config.json', 'pytorch_model.bin', 'special_tokens_map.json', 'tokenizer_config.json']
    
    for file in required_files:
        if not any(file in f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))):
            return False
    
    return True


def clear_model_cache(model_type=None):
    """
    Очищает кэш моделей.
    
    Args:
        model_type (str, optional): Тип модели для очистки ('layoutlm', 'donut' или None для всех)
        
    Returns:
        bool: True, если кэш успешно очищен, иначе False
    """
    try:
        if model_type:
            model_dir = os.path.join(config.MODELS_PATH, model_type.lower())
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
        else:
            # Очищаем весь кэш моделей
            if os.path.exists(config.MODELS_PATH):
                for item in os.listdir(config.MODELS_PATH):
                    item_path = os.path.join(config.MODELS_PATH, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
        
        return True
    except Exception as e:
        print(f"Ошибка при очистке кэша моделей: {str(e)}")
        return False


def format_size(size_bytes):
    """
    Форматирует размер из байтов в человекочитаемый формат.
    
    Args:
        size_bytes (int): Размер в байтах
        
    Returns:
        str: Форматированный размер (например, '15.5 MB')
    """
    if size_bytes == 0:
        return "0 B"
    
    # Определяем единицы измерения
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    
    # Вычисляем логарифм с основанием 1024
    import math
    if size_bytes == 0:
        i = 0
    else:
        i = int(round(math.log(size_bytes) / math.log(1024)))
    
    # Округляем до двух знаков после запятой
    p = 1024 ** i
    s = round(size_bytes / p, 2)
    
    return f"{s} {units[i]}"


def format_timestamp(timestamp):
    """
    Форматирует временную метку в читаемый формат.
    
    Args:
        timestamp (float): Временная метка в секундах с начала эпохи
        
    Returns:
        str: Отформатированная дата и время
    """
    return datetime.datetime.fromtimestamp(timestamp).strftime("%d.%m.%Y %H:%M:%S")


def get_model_info_formatted(model_type):
    """
    Получает отформатированную информацию о модели.
    
    Args:
        model_type (str): Тип модели ('layoutlm' или 'donut')
        
    Returns:
        str: Отформатированная информация о модели
    """
    models_info = get_model_cache_info()
    
    if model_type.lower() not in models_info:
        return "Модель не найдена в кэше"
    
    info = models_info[model_type.lower()]
    
    # Форматируем информацию
    size_formatted = format_size(info['size'])
    date_formatted = format_timestamp(info['last_modified'])
    status = "Полная" if info['is_complete'] else "Неполная"
    
    # Возвращаем отформатированную строку
    model_id = config.LAYOUTLM_MODEL_ID if model_type.lower() == 'layoutlm' else config.DONUT_MODEL_ID
    
    return f"ID модели: {model_id}\nРазмер: {size_formatted}\nПоследнее обновление: {date_formatted}\nСтатус: {status}" 