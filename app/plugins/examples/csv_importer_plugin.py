"""
CSV Importer Plugin для InvoiceGemini
Плагин для импорта данных из CSV файлов
"""
import csv
from typing import Dict, Any, List
from pathlib import Path
import pandas as pd

from ..base_plugin import ImporterPlugin, PluginMetadata, PluginType, PluginCapability, PluginStatus


class CSVImporterPlugin(ImporterPlugin):
    """Плагин для импорта данных из CSV файлов"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.encoding = config.get('encoding', 'utf-8') if config else 'utf-8'
        self.delimiter = config.get('delimiter', ',') if config else ','
        self.has_header = config.get('has_header', True) if config else True
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="CSV Importer",
            version="1.0.0",
            description="Импорт данных из CSV файлов с поддержкой различных кодировок",
            author="InvoiceGemini Team",
            plugin_type=PluginType.IMPORTER,
            capabilities=[PluginCapability.TEXT, PluginCapability.BATCH],
            supported_formats=['csv', 'tsv'],
            config_schema={
                "required": [],
                "types": {
                    "encoding": str,
                    "delimiter": str,
                    "has_header": bool
                }
            }
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        try:
            # Проверяем наличие pandas
            import pandas as pd
            self.status = PluginStatus.LOADED
            return True
        except ImportError:
            self.set_error("Pandas не установлен")
            return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        pass
    
    def import_data(self, source: str, **kwargs) -> Dict[str, Any]:
        """
        Импорт данных из CSV файла
        
        Args:
            source: Путь к CSV файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict с импортированными данными
        """
        try:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Файл не найден: {source}")
            
            # Определяем разделитель по расширению
            delimiter = self.delimiter
            if source_path.suffix.lower() == '.tsv':
                delimiter = '\t'
            
            # Читаем CSV с помощью pandas
            df = pd.read_csv(
                source_path,
                encoding=self.encoding,
                delimiter=delimiter,
                header=0 if self.has_header else None
            )
            
            # Преобразуем в словарь
            records = df.to_dict('records')
            
            return {
                'success': True,
                'data': records,
                'total_records': len(records),
                'columns': list(df.columns),
                'source_file': str(source_path),
                'metadata': {
                    'encoding': self.encoding,
                    'delimiter': delimiter,
                    'has_header': self.has_header,
                    'shape': df.shape
                }
            }
            
        except Exception as e:
            self.set_error(str(e))
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def get_supported_sources(self) -> List[str]:
        """Возвращает список поддерживаемых источников"""
        return ['csv', 'tsv', 'file']
    
    def validate_source(self, source: str) -> bool:
        """Валидация источника данных"""
        try:
            source_path = Path(source)
            if not source_path.exists():
                return False
            
            # Проверяем расширение файла
            supported_extensions = ['.csv', '.tsv']
            return source_path.suffix.lower() in supported_extensions
            
        except Exception:
            return False
    
    def get_preview(self, source: str, max_rows: int = 10) -> Dict[str, Any]:
        """
        Получает предварительный просмотр данных
        
        Args:
            source: Путь к файлу
            max_rows: Максимальное количество строк
            
        Returns:
            Словарь с предварительными данными
        """
        try:
            source_path = Path(source)
            delimiter = self.delimiter
            if source_path.suffix.lower() == '.tsv':
                delimiter = '\t'
            
            # Читаем только первые строки
            df = pd.read_csv(
                source_path,
                encoding=self.encoding,
                delimiter=delimiter,
                header=0 if self.has_header else None,
                nrows=max_rows
            )
            
            return {
                'success': True,
                'preview_data': df.to_dict('records'),
                'columns': list(df.columns),
                'total_preview_rows': len(df)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


class ExcelImporterPlugin(ImporterPlugin):
    """Плагин для импорта данных из Excel файлов"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.sheet_name = config.get('sheet_name', 0) if config else 0
        self.header_row = config.get('header_row', 0) if config else 0
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Excel Importer",
            version="1.0.0",
            description="Импорт данных из Excel файлов (XLS, XLSX)",
            author="InvoiceGemini Team",
            plugin_type=PluginType.IMPORTER,
            capabilities=[PluginCapability.TEXT, PluginCapability.BATCH],
            supported_formats=['xlsx', 'xls'],
            config_schema={
                "required": [],
                "types": {
                    "sheet_name": [str, int],
                    "header_row": int
                }
            }
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        try:
            import pandas as pd
            import openpyxl  # Для xlsx
            self.status = PluginStatus.LOADED
            return True
        except ImportError as e:
            self.set_error(f"Требуемые библиотеки не установлены: {e}")
            return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        pass
    
    def import_data(self, source: str, **kwargs) -> Dict[str, Any]:
        """
        Импорт данных из Excel файла
        
        Args:
            source: Путь к Excel файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            Dict с импортированными данными
        """
        try:
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"Файл не найден: {source}")
            
            # Читаем Excel файл
            df = pd.read_excel(
                source_path,
                sheet_name=self.sheet_name,
                header=self.header_row
            )
            
            # Преобразуем NaN в None для JSON совместимости
            df = df.where(pd.notna(df), None)
            
            # Преобразуем в словарь
            records = df.to_dict('records')
            
            # Получаем информацию о листах
            excel_file = pd.ExcelFile(source_path)
            sheet_names = excel_file.sheet_names
            
            return {
                'success': True,
                'data': records,
                'total_records': len(records),
                'columns': list(df.columns),
                'source_file': str(source_path),
                'metadata': {
                    'sheet_name': self.sheet_name,
                    'header_row': self.header_row,
                    'shape': df.shape,
                    'available_sheets': sheet_names
                }
            }
            
        except Exception as e:
            self.set_error(str(e))
            return {
                'success': False,
                'error': str(e),
                'data': []
            }
    
    def get_supported_sources(self) -> List[str]:
        """Возвращает список поддерживаемых источников"""
        return ['xlsx', 'xls', 'file']
    
    def validate_source(self, source: str) -> bool:
        """Валидация источника данных"""
        try:
            source_path = Path(source)
            if not source_path.exists():
                return False
            
            # Проверяем расширение файла
            supported_extensions = ['.xlsx', '.xls']
            return source_path.suffix.lower() in supported_extensions
            
        except Exception:
            return False
    
    def get_sheet_names(self, source: str) -> List[str]:
        """
        Получает список названий листов в Excel файле
        
        Args:
            source: Путь к файлу
            
        Returns:
            Список названий листов
        """
        try:
            excel_file = pd.ExcelFile(Path(source))
            return excel_file.sheet_names
        except Exception as e:
            return [] 