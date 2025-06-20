"""
Универсальный менеджер плагинов для InvoiceGemini
Поддерживает все типы плагинов: LLM, обработка, просмотр, экспорт, валидация, трансформация
"""
import os
import importlib
import importlib.util
import json
from typing import Dict, List, Type, Optional, Any, Union
from pathlib import Path
from .base_plugin import (
    BasePlugin, PluginType, PluginCapability, PluginMetadata,
    LLMPlugin, ViewerPlugin, ExporterPlugin, ValidatorPlugin, 
    TransformerPlugin, ProcessorPlugin
)

class UniversalPluginManager:
    """
    Универсальный менеджер для управления всеми типами плагинов
    """
    
    def __init__(self, plugins_dir: str = None):
        """
        Инициализация универсального менеджера плагинов
        
        Args:
            plugins_dir: Директория с пользовательскими плагинами
        """
        self.builtin_plugins_dir = os.path.join(os.path.dirname(__file__), "models")
        self.user_plugins_dir = plugins_dir or os.path.join(os.getcwd(), "plugins", "user")
        
        # Реестры плагинов по типам
        self.plugin_classes: Dict[PluginType, Dict[str, Type[BasePlugin]]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        self.plugin_instances: Dict[PluginType, Dict[str, BasePlugin]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        
        # Создаем директории
        os.makedirs(self.user_plugins_dir, exist_ok=True)
        
        print(f"🔧 Инициализация UniversalPluginManager...")
        print(f"📁 Встроенные плагины: {self.builtin_plugins_dir}")
        print(f"📁 Пользовательские плагины: {self.user_plugins_dir}")
        
        # Загружаем все доступные плагины
        self._load_all_plugins()
    
    def _load_all_plugins(self):
        """Загружает все доступные плагины всех типов"""
        print("🔄 Загрузка плагинов...")
        
        # Загружаем встроенные плагины
        self._load_builtin_plugins()
        
        # Загружаем пользовательские плагины
        self._load_user_plugins()
        
        # Выводим статистику
        total_plugins = sum(len(plugins) for plugins in self.plugin_classes.values())
        print(f"✅ Загружено плагинов: {total_plugins}")
        
        for plugin_type, plugins in self.plugin_classes.items():
            if plugins:
                print(f"   📋 {plugin_type.value}: {list(plugins.keys())}")
    
    def _load_builtin_plugins(self):
        """Загружает встроенные плагины"""
        # LLM плагины
        llm_plugins = [
            ("gemini_plugin", "GeminiPlugin"),
            ("openai_plugin", "OpenAIPlugin"),
            ("anthropic_plugin", "AnthropicPlugin"),
            ("universal_llm_plugin", "UniversalLLMPlugin"),
            ("llama_plugin", "LlamaPlugin"),
            ("mistral_plugin", "MistralPlugin"),
            ("codellama_plugin", "CodeLlamaPlugin"),
        ]
        
        for module_name, class_name in llm_plugins:
            self._load_builtin_plugin(module_name, class_name, PluginType.LLM)
        
        # Встроенные экспортеры
        self._load_builtin_exporters()
        
        # Встроенные просмотрщики
        self._load_builtin_viewers()
        
        # Встроенные валидаторы
        self._load_builtin_validators()
    
    def _load_builtin_plugin(self, module_name: str, class_name: str, plugin_type: PluginType):
        """Загружает встроенный плагин"""
        try:
            module_path = f"app.plugins.models.{module_name}"
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            
            # Проверяем соответствие типу
            if self._is_plugin_type(plugin_class, plugin_type):
                plugin_id = class_name.lower().replace("plugin", "")
                self.plugin_classes[plugin_type][plugin_id] = plugin_class
                print(f"✅ Загружен встроенный {plugin_type.value} плагин: {class_name}")
            
        except ImportError as e:
            print(f"⚠️ Встроенный плагин {class_name} не найден: {e}")
        except Exception as e:
            print(f"❌ Ошибка загрузки встроенного плагина {class_name}: {e}")
    
    def _load_builtin_exporters(self):
        """Загружает встроенные экспортеры"""
        # Создаем встроенные экспортеры
        self._create_json_exporter()
        self._create_excel_exporter()
        self._create_csv_exporter()
        self._create_pdf_exporter()
    
    def _load_builtin_viewers(self):
        """Загружает встроенные просмотрщики"""
        # Создаем встроенные просмотрщики
        self._create_table_viewer()
        self._create_preview_viewer()
    
    def _load_builtin_validators(self):
        """Загружает встроенные валидаторы"""
        # Создаем встроенные валидаторы
        self._create_invoice_validator()
        self._create_data_validator()
    
    def _create_json_exporter(self):
        """Создает встроенный JSON экспортер"""
        class JSONExporter(ExporterPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="JSON Exporter",
                    version="1.0.0",
                    description="Экспорт данных в формат JSON",
                    author="InvoiceGemini Team",
                    plugin_type=PluginType.EXPORTER,
                    capabilities=[PluginCapability.TEXT]
                )
            
            def initialize(self) -> bool:
                self.is_loaded = True
                return True
            
            def cleanup(self):
                pass
            
            def export(self, data: Any, output_path: str, **kwargs) -> bool:
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    return True
                except Exception as e:
                    print(f"Ошибка экспорта JSON: {e}")
                    return False
            
            def get_supported_formats(self) -> List[str]:
                return ["json"]
        
        self.plugin_classes[PluginType.EXPORTER]["json"] = JSONExporter
    
    def _create_excel_exporter(self):
        """Создает встроенный Excel экспортер"""
        class ExcelExporter(ExporterPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="Excel Exporter",
                    version="1.0.0",
                    description="Экспорт данных в формат Excel",
                    author="InvoiceGemini Team",
                    plugin_type=PluginType.EXPORTER,
                    capabilities=[PluginCapability.TEXT],
                    dependencies=["openpyxl"]
                )
            
            def initialize(self) -> bool:
                try:
                    import openpyxl
                    self.is_loaded = True
                    return True
                except ImportError:
                    print("⚠️ Для Excel экспорта требуется openpyxl")
                    return False
            
            def cleanup(self):
                pass
            
            def export(self, data: Any, output_path: str, **kwargs) -> bool:
                try:
                    import openpyxl
                    from openpyxl import Workbook
                    
                    wb = Workbook()
                    ws = wb.active
                    ws.title = "Invoice Data"
                    
                    if isinstance(data, list):
                        # Пакетные данные
                        if data and isinstance(data[0], dict):
                            headers = list(data[0].keys())
                            ws.append(headers)
                            for row_data in data:
                                ws.append([row_data.get(h, "") for h in headers])
                    elif isinstance(data, dict):
                        # Одиночные данные
                        for key, value in data.items():
                            ws.append([key, str(value)])
                    
                    wb.save(output_path)
                    return True
                except Exception as e:
                    print(f"Ошибка экспорта Excel: {e}")
                    return False
            
            def get_supported_formats(self) -> List[str]:
                return ["xlsx", "xls"]
        
        self.plugin_classes[PluginType.EXPORTER]["excel"] = ExcelExporter
    
    def _create_csv_exporter(self):
        """Создает встроенный CSV экспортер"""
        class CSVExporter(ExporterPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="CSV Exporter",
                    version="1.0.0",
                    description="Экспорт данных в формат CSV",
                    author="InvoiceGemini Team",
                    plugin_type=PluginType.EXPORTER,
                    capabilities=[PluginCapability.TEXT]
                )
            
            def initialize(self) -> bool:
                self.is_loaded = True
                return True
            
            def cleanup(self):
                pass
            
            def export(self, data: Any, output_path: str, **kwargs) -> bool:
                try:
                    import csv
                    
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        if isinstance(data, list) and data and isinstance(data[0], dict):
                            # Пакетные данные
                            headers = list(data[0].keys())
                            writer = csv.DictWriter(f, fieldnames=headers)
                            writer.writeheader()
                            writer.writerows(data)
                        elif isinstance(data, dict):
                            # Одиночные данные
                            writer = csv.writer(f)
                            for key, value in data.items():
                                writer.writerow([key, str(value)])
                    
                    return True
                except Exception as e:
                    print(f"Ошибка экспорта CSV: {e}")
                    return False
            
            def get_supported_formats(self) -> List[str]:
                return ["csv"]
        
        self.plugin_classes[PluginType.EXPORTER]["csv"] = CSVExporter
    
    def _create_pdf_exporter(self):
        """Создает встроенный PDF экспортер"""
        class PDFExporter(ExporterPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="PDF Exporter",
                    version="1.0.0",
                    description="Экспорт данных в формат PDF",
                    author="InvoiceGemini Team",
                    plugin_type=PluginType.EXPORTER,
                    capabilities=[PluginCapability.TEXT],
                    dependencies=["reportlab"]
                )
            
            def initialize(self) -> bool:
                try:
                    import reportlab
                    self.is_loaded = True
                    return True
                except ImportError:
                    print("⚠️ Для PDF экспорта требуется reportlab")
                    return False
            
            def cleanup(self):
                pass
            
            def export(self, data: Any, output_path: str, **kwargs) -> bool:
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
                    from reportlab.lib import colors
                    
                    doc = SimpleDocTemplate(output_path, pagesize=letter)
                    elements = []
                    
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        # Пакетные данные
                        headers = list(data[0].keys())
                        table_data = [headers]
                        for row_data in data:
                            table_data.append([str(row_data.get(h, "")) for h in headers])
                    elif isinstance(data, dict):
                        # Одиночные данные
                        table_data = [[key, str(value)] for key, value in data.items()]
                    else:
                        table_data = [["Data", str(data)]]
                    
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    
                    elements.append(table)
                    doc.build(elements)
                    return True
                except Exception as e:
                    print(f"Ошибка экспорта PDF: {e}")
                    return False
            
            def get_supported_formats(self) -> List[str]:
                return ["pdf"]
        
        self.plugin_classes[PluginType.EXPORTER]["pdf"] = PDFExporter
    
    def _create_table_viewer(self):
        """Создает встроенный табличный просмотрщик"""
        class TableViewer(ViewerPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="Table Viewer",
                    version="1.0.0",
                    description="Табличный просмотр данных",
                    author="InvoiceGemini Team",
                    plugin_type=PluginType.VIEWER,
                    capabilities=[PluginCapability.TEXT]
                )
            
            def initialize(self) -> bool:
                self.is_loaded = True
                return True
            
            def cleanup(self):
                pass
            
            def create_viewer(self, data: Any, parent=None) -> Any:
                try:
                    from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
                    
                    table = QTableWidget(parent)
                    self.update_view(table, data)
                    return table
                except Exception as e:
                    print(f"Ошибка создания табличного просмотрщика: {e}")
                    return None
            
            def update_view(self, viewer: Any, data: Any):
                try:
                    from PyQt6.QtWidgets import QTableWidgetItem
                    
                    if isinstance(data, dict):
                        # Одиночные данные
                        viewer.setRowCount(len(data))
                        viewer.setColumnCount(2)
                        viewer.setHorizontalHeaderLabels(["Поле", "Значение"])
                        
                        for row, (key, value) in enumerate(data.items()):
                            viewer.setItem(row, 0, QTableWidgetItem(str(key)))
                            viewer.setItem(row, 1, QTableWidgetItem(str(value)))
                    
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        # Пакетные данные
                        headers = list(data[0].keys())
                        viewer.setRowCount(len(data))
                        viewer.setColumnCount(len(headers))
                        viewer.setHorizontalHeaderLabels(headers)
                        
                        for row, row_data in enumerate(data):
                            for col, header in enumerate(headers):
                                value = row_data.get(header, "")
                                viewer.setItem(row, col, QTableWidgetItem(str(value)))
                    
                except Exception as e:
                    print(f"Ошибка обновления табличного просмотрщика: {e}")
        
        self.plugin_classes[PluginType.VIEWER]["table"] = TableViewer
    
    def _create_preview_viewer(self):
        """Создает встроенный просмотрщик предварительного просмотра"""
        class PreviewViewer(ViewerPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="Preview Viewer",
                    version="1.0.0",
                    description="Предварительный просмотр данных",
                    author="InvoiceGemini Team",
                    plugin_type=PluginType.VIEWER,
                    capabilities=[PluginCapability.TEXT]
                )
            
            def initialize(self) -> bool:
                self.is_loaded = True
                return True
            
            def cleanup(self):
                pass
            
            def create_viewer(self, data: Any, parent=None) -> Any:
                try:
                    from PyQt6.QtWidgets import QTextEdit
                    
                    text_edit = QTextEdit(parent)
                    text_edit.setReadOnly(True)
                    self.update_view(text_edit, data)
                    return text_edit
                except Exception as e:
                    print(f"Ошибка создания просмотрщика предварительного просмотра: {e}")
                    return None
            
            def update_view(self, viewer: Any, data: Any):
                try:
                    if isinstance(data, dict):
                        text = json.dumps(data, indent=2, ensure_ascii=False)
                    elif isinstance(data, list):
                        text = json.dumps(data, indent=2, ensure_ascii=False)
                    else:
                        text = str(data)
                    
                    viewer.setPlainText(text)
                except Exception as e:
                    print(f"Ошибка обновления просмотрщика предварительного просмотра: {e}")
        
        self.plugin_classes[PluginType.VIEWER]["preview"] = PreviewViewer
    
    def _create_invoice_validator(self):
        """Создает встроенный валидатор счетов"""
        class InvoiceValidator(ValidatorPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="Invoice Validator",
                    version="1.0.0",
                    description="Валидация данных счетов",
                    author="InvoiceGemini Team",
                    plugin_type=PluginType.VALIDATOR,
                    capabilities=[PluginCapability.TEXT]
                )
            
            def initialize(self) -> bool:
                self.is_loaded = True
                return True
            
            def cleanup(self):
                pass
            
            def validate(self, data: Any) -> Dict[str, Any]:
                errors = []
                warnings = []
                
                if not isinstance(data, dict):
                    errors.append("Данные должны быть в формате словаря")
                    return {"errors": errors, "warnings": warnings}
                
                # Проверяем обязательные поля
                required_fields = ["sender", "invoice_number", "total"]
                for field in required_fields:
                    if field not in data or not data[field]:
                        errors.append(f"Отсутствует обязательное поле: {field}")
                
                # Проверяем числовые поля
                numeric_fields = ["total", "amount_no_vat", "vat_percent"]
                for field in numeric_fields:
                    if field in data and data[field]:
                        try:
                            float(str(data[field]).replace(",", "."))
                        except ValueError:
                            errors.append(f"Поле {field} должно содержать числовое значение")
                
                # Проверяем дату
                if "invoice_date" in data and data["invoice_date"]:
                    import re
                    date_pattern = r'\d{1,2}[./]\d{1,2}[./]\d{2,4}'
                    if not re.match(date_pattern, str(data["invoice_date"])):
                        warnings.append("Формат даты может быть некорректным")
                
                return {"errors": errors, "warnings": warnings}
        
        self.plugin_classes[PluginType.VALIDATOR]["invoice"] = InvoiceValidator
    
    def _create_data_validator(self):
        """Создает встроенный валидатор данных"""
        class DataValidator(ValidatorPlugin):
            @property
            def metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name="Data Validator",
                    version="1.0.0",
                    description="Общая валидация данных",
                    author="InvoiceGemini Team",
                    plugin_type=PluginType.VALIDATOR,
                    capabilities=[PluginCapability.TEXT]
                )
            
            def initialize(self) -> bool:
                self.is_loaded = True
                return True
            
            def cleanup(self):
                pass
            
            def validate(self, data: Any) -> Dict[str, Any]:
                errors = []
                warnings = []
                
                if data is None:
                    errors.append("Данные отсутствуют")
                elif isinstance(data, dict) and not data:
                    warnings.append("Словарь данных пуст")
                elif isinstance(data, list) and not data:
                    warnings.append("Список данных пуст")
                
                return {"errors": errors, "warnings": warnings}
        
        self.plugin_classes[PluginType.VALIDATOR]["data"] = DataValidator
    
    def _load_user_plugins(self):
        """Загружает пользовательские плагины"""
        if not os.path.exists(self.user_plugins_dir):
            return
        
        for filename in os.listdir(self.user_plugins_dir):
            if filename.endswith('_plugin.py') and not filename.startswith('__'):
                self._load_plugin_file(filename, self.user_plugins_dir)
    
    def _load_plugin_file(self, filename: str, plugins_dir: str):
        """Загружает плагин из файла"""
        try:
            module_name = filename[:-3]
            file_path = os.path.join(plugins_dir, filename)
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Ищем классы плагинов
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, BasePlugin) and attr != BasePlugin:
                    plugin_type = self._detect_plugin_type(attr)
                    if plugin_type:
                        plugin_id = attr_name.lower().replace("plugin", "")
                        self.plugin_classes[plugin_type][plugin_id] = attr
                        print(f"✅ Загружен пользовательский {plugin_type.value} плагин: {attr_name}")
                        
        except Exception as e:
            print(f"❌ Ошибка загрузки плагина {filename}: {e}")
    
    def _is_plugin_type(self, plugin_class: Type, plugin_type: PluginType) -> bool:
        """Проверяет соответствие класса плагина типу"""
        type_mapping = {
            PluginType.LLM: LLMPlugin,
            PluginType.VIEWER: ViewerPlugin,
            PluginType.EXPORTER: ExporterPlugin,
            PluginType.VALIDATOR: ValidatorPlugin,
            PluginType.TRANSFORMER: TransformerPlugin,
            PluginType.PROCESSOR: ProcessorPlugin
        }
        
        base_class = type_mapping.get(plugin_type, BasePlugin)
        return issubclass(plugin_class, base_class)
    
    def _detect_plugin_type(self, plugin_class: Type) -> Optional[PluginType]:
        """Определяет тип плагина по классу"""
        if issubclass(plugin_class, LLMPlugin):
            return PluginType.LLM
        elif issubclass(plugin_class, ViewerPlugin):
            return PluginType.VIEWER
        elif issubclass(plugin_class, ExporterPlugin):
            return PluginType.EXPORTER
        elif issubclass(plugin_class, ValidatorPlugin):
            return PluginType.VALIDATOR
        elif issubclass(plugin_class, TransformerPlugin):
            return PluginType.TRANSFORMER
        elif issubclass(plugin_class, ProcessorPlugin):
            return PluginType.PROCESSOR
        return None
    
    # Публичные методы для работы с плагинами
    
    def get_available_plugins(self, plugin_type: PluginType = None) -> Dict[str, List[str]]:
        """Возвращает список доступных плагинов"""
        if plugin_type:
            return {plugin_type.value: list(self.plugin_classes[plugin_type].keys())}
        
        return {
            plugin_type.value: list(plugins.keys())
            for plugin_type, plugins in self.plugin_classes.items()
            if plugins
        }
    
    def create_plugin_instance(self, plugin_type: PluginType, plugin_id: str, **kwargs) -> Optional[BasePlugin]:
        """Создает экземпляр плагина"""
        if plugin_id not in self.plugin_classes[plugin_type]:
            print(f"❌ Плагин {plugin_id} типа {plugin_type.value} не найден")
            return None
        
        try:
            if plugin_id not in self.plugin_instances[plugin_type]:
                plugin_class = self.plugin_classes[plugin_type][plugin_id]
                instance = plugin_class(**kwargs)
                
                if instance.initialize():
                    self.plugin_instances[plugin_type][plugin_id] = instance
                    print(f"✅ Создан экземпляр {plugin_type.value} плагина: {plugin_id}")
                else:
                    print(f"❌ Не удалось инициализировать плагин {plugin_id}")
                    return None
            
            return self.plugin_instances[plugin_type][plugin_id]
            
        except Exception as e:
            print(f"❌ Ошибка создания экземпляра плагина {plugin_id}: {e}")
            return None
    
    def get_plugin_instance(self, plugin_type: PluginType, plugin_id: str) -> Optional[BasePlugin]:
        """Возвращает существующий экземпляр плагина"""
        return self.plugin_instances[plugin_type].get(plugin_id)
    
    def remove_plugin_instance(self, plugin_type: PluginType, plugin_id: str) -> bool:
        """Удаляет экземпляр плагина"""
        try:
            if plugin_id in self.plugin_instances[plugin_type]:
                instance = self.plugin_instances[plugin_type][plugin_id]
                instance.cleanup()
                del self.plugin_instances[plugin_type][plugin_id]
                print(f"✅ Удален экземпляр {plugin_type.value} плагина: {plugin_id}")
                return True
            return False
        except Exception as e:
            print(f"❌ Ошибка удаления экземпляра плагина {plugin_id}: {e}")
            return False
    
    def get_plugin_info(self, plugin_type: PluginType, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает информацию о плагине"""
        if plugin_id not in self.plugin_classes[plugin_type]:
            return None
        
        plugin_class = self.plugin_classes[plugin_type][plugin_id]
        instance = self.plugin_instances[plugin_type].get(plugin_id)
        
        # Создаем временный экземпляр для получения метаданных
        try:
            temp_instance = plugin_class()
            metadata = temp_instance.metadata.to_dict()
            temp_instance.cleanup()
        except (TypeError, AttributeError, ImportError, Exception) as e:
            # Ошибка создания временного экземпляра - используем базовые метаданные
            metadata = {"name": plugin_id, "type": plugin_type.value}
        
        return {
            **metadata,
            "id": plugin_id,
            "is_loaded": instance is not None,
            "is_enabled": instance.is_enabled if instance else True
        }
    
    def export_data(self, data: Any, output_path: str, format_type: str, **kwargs) -> bool:
        """Экспортирует данные используя подходящий плагин"""
        # Ищем подходящий экспортер
        for plugin_id, plugin_class in self.plugin_classes[PluginType.EXPORTER].items():
            instance = self.get_plugin_instance(PluginType.EXPORTER, plugin_id)
            if not instance:
                instance = self.create_plugin_instance(PluginType.EXPORTER, plugin_id)
            
            if instance and format_type in instance.get_supported_formats():
                return instance.export(data, output_path, **kwargs)
        
        print(f"❌ Не найден экспортер для формата {format_type}")
        return False
    
    def validate_data(self, data: Any, validator_type: str = "data") -> Dict[str, Any]:
        """Валидирует данные используя подходящий валидатор"""
        instance = self.get_plugin_instance(PluginType.VALIDATOR, validator_type)
        if not instance:
            instance = self.create_plugin_instance(PluginType.VALIDATOR, validator_type)
        
        if instance:
            return instance.validate(data)
        
        return {"errors": [f"Валидатор {validator_type} не найден"], "warnings": []}
    
    def create_viewer(self, data: Any, viewer_type: str = "table", parent=None) -> Any:
        """Создает просмотрщик данных"""
        instance = self.get_plugin_instance(PluginType.VIEWER, viewer_type)
        if not instance:
            instance = self.create_plugin_instance(PluginType.VIEWER, viewer_type)
        
        if instance:
            return instance.create_viewer(data, parent)
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по плагинам"""
        stats = {}
        total_classes = 0
        total_instances = 0
        
        for plugin_type in PluginType:
            type_classes = len(self.plugin_classes[plugin_type])
            type_instances = len(self.plugin_instances[plugin_type])
            
            stats[plugin_type.value] = {
                "available": type_classes,
                "loaded": type_instances,
                "plugins": list(self.plugin_classes[plugin_type].keys())
            }
            
            total_classes += type_classes
            total_instances += type_instances
        
        stats["total"] = {
            "available": total_classes,
            "loaded": total_instances
        }
        
        return stats
    
    def get_plugin_statistics(self) -> Dict[str, int]:
        """Возвращает упрощенную статистику по плагинам (для совместимости)"""
        stats = {}
        for plugin_type in PluginType:
            count = len(self.plugin_classes[plugin_type])
            if count > 0:
                stats[plugin_type.value] = count
        return stats
    
    def get_plugin(self, plugin_type_str: str, plugin_id: str) -> Optional[BasePlugin]:
        """Получает экземпляр плагина по строковому типу и ID"""
        try:
            # Преобразуем строку в PluginType
            plugin_type = PluginType(plugin_type_str)
            
            # Сначала пытаемся получить существующий экземпляр
            instance = self.get_plugin_instance(plugin_type, plugin_id)
            if instance:
                return instance
            
            # Если экземпляра нет, создаем новый
            return self.create_plugin_instance(plugin_type, plugin_id)
            
        except ValueError:
            print(f"❌ Неизвестный тип плагина: {plugin_type_str}")
            return None
        except Exception as e:
            print(f"❌ Ошибка получения плагина {plugin_id}: {e}")
            return None 