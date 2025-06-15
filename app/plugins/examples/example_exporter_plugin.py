"""
Пример плагина экспорта для демонстрации универсальной системы плагинов InvoiceGemini
"""
from typing import Dict, Any, List
import json
import xml.etree.ElementTree as ET
import sys
import os

# Добавляем путь к родительскому модулю для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from base_plugin import ExporterPlugin, PluginMetadata, PluginType, PluginCapability
except ImportError:
    # Альтернативный импорт для тестирования
    from app.plugins.base_plugin import ExporterPlugin, PluginMetadata, PluginType, PluginCapability

class XMLExporterPlugin(ExporterPlugin):
    """
    Пример плагина для экспорта данных в формат XML
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="XML Exporter",
            version="1.0.0",
            description="Экспорт данных счетов в формат XML",
            author="InvoiceGemini Team",
            plugin_type=PluginType.EXPORTER,
            capabilities=[PluginCapability.TEXT],
            dependencies=[],
            config_schema={
                "required": [],
                "optional": {
                    "encoding": "utf-8",
                    "pretty_print": True,
                    "root_element": "invoices"
                }
            }
        )
    
    def initialize(self) -> bool:
        """Инициализирует плагин"""
        self.is_loaded = True
        print("✅ XML Exporter Plugin инициализирован")
        return True
    
    def cleanup(self):
        """Очищает ресурсы плагина"""
        self.is_loaded = False
        print("🧹 XML Exporter Plugin очищен")
    
    def export(self, data: Any, output_path: str, **kwargs) -> bool:
        """
        Экспортирует данные в XML формат
        
        Args:
            data: Данные для экспорта (dict или list)
            output_path: Путь к выходному файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            bool: True если экспорт успешен
        """
        try:
            # Получаем настройки из конфигурации
            encoding = self.config.get('encoding', 'utf-8')
            pretty_print = self.config.get('pretty_print', True)
            root_element = self.config.get('root_element', 'invoices')
            
            # Создаем корневой элемент
            root = ET.Element(root_element)
            
            if isinstance(data, list):
                # Пакетные данные
                for i, item in enumerate(data):
                    invoice_elem = ET.SubElement(root, "invoice")
                    invoice_elem.set("id", str(i + 1))
                    self._dict_to_xml(item, invoice_elem)
            elif isinstance(data, dict):
                # Одиночные данные
                invoice_elem = ET.SubElement(root, "invoice")
                self._dict_to_xml(data, invoice_elem)
            else:
                # Простые данные
                data_elem = ET.SubElement(root, "data")
                data_elem.text = str(data)
            
            # Создаем дерево XML
            tree = ET.ElementTree(root)
            
            # Сохраняем в файл
            if pretty_print:
                self._prettify_xml(root)
            
            tree.write(output_path, encoding=encoding, xml_declaration=True)
            
            print(f"✅ Данные успешно экспортированы в XML: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка экспорта в XML: {e}")
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Возвращает список поддерживаемых форматов"""
        return ["xml"]
    
    def _dict_to_xml(self, data_dict: Dict[str, Any], parent_element: ET.Element):
        """
        Преобразует словарь в XML элементы
        
        Args:
            data_dict: Словарь данных
            parent_element: Родительский XML элемент
        """
        for key, value in data_dict.items():
            # Очищаем ключ от недопустимых символов для XML
            clean_key = self._clean_xml_key(key)
            
            if isinstance(value, dict):
                # Вложенный словарь
                child_elem = ET.SubElement(parent_element, clean_key)
                self._dict_to_xml(value, child_elem)
            elif isinstance(value, list):
                # Список значений
                for i, item in enumerate(value):
                    item_elem = ET.SubElement(parent_element, clean_key)
                    item_elem.set("index", str(i))
                    if isinstance(item, dict):
                        self._dict_to_xml(item, item_elem)
                    else:
                        item_elem.text = str(item)
            else:
                # Простое значение
                child_elem = ET.SubElement(parent_element, clean_key)
                child_elem.text = str(value) if value is not None else ""
    
    def _clean_xml_key(self, key: str) -> str:
        """
        Очищает ключ для использования в XML
        
        Args:
            key: Исходный ключ
            
        Returns:
            str: Очищенный ключ
        """
        # Заменяем пробелы и недопустимые символы
        clean_key = key.replace(" ", "_").replace("-", "_")
        clean_key = "".join(c for c in clean_key if c.isalnum() or c == "_")
        
        # Убеждаемся, что ключ начинается с буквы
        if clean_key and not clean_key[0].isalpha():
            clean_key = "field_" + clean_key
        
        return clean_key or "unknown_field"
    
    def _prettify_xml(self, element: ET.Element, level: int = 0):
        """
        Добавляет отступы для красивого форматирования XML
        
        Args:
            element: XML элемент
            level: Уровень вложенности
        """
        indent = "\n" + level * "  "
        if len(element):
            if not element.text or not element.text.strip():
                element.text = indent + "  "
            if not element.tail or not element.tail.strip():
                element.tail = indent
            for child in element:
                self._prettify_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not element.tail or not element.tail.strip()):
                element.tail = indent

class YAMLExporterPlugin(ExporterPlugin):
    """
    Пример плагина для экспорта данных в формат YAML
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="YAML Exporter",
            version="1.0.0",
            description="Экспорт данных счетов в формат YAML",
            author="InvoiceGemini Team",
            plugin_type=PluginType.EXPORTER,
            capabilities=[PluginCapability.TEXT],
            dependencies=["pyyaml"],
            config_schema={
                "required": [],
                "optional": {
                    "encoding": "utf-8",
                    "default_flow_style": False,
                    "allow_unicode": True
                }
            }
        )
    
    def initialize(self) -> bool:
        """Инициализирует плагин"""
        try:
            import yaml
            self.yaml = yaml
            self.is_loaded = True
            print("✅ YAML Exporter Plugin инициализирован")
            return True
        except ImportError:
            print("❌ PyYAML не установлен. Установите: pip install pyyaml")
            return False
    
    def cleanup(self):
        """Очищает ресурсы плагина"""
        self.is_loaded = False
        print("🧹 YAML Exporter Plugin очищен")
    
    def export(self, data: Any, output_path: str, **kwargs) -> bool:
        """
        Экспортирует данные в YAML формат
        
        Args:
            data: Данные для экспорта
            output_path: Путь к выходному файлу
            **kwargs: Дополнительные параметры
            
        Returns:
            bool: True если экспорт успешен
        """
        try:
            # Получаем настройки из конфигурации
            encoding = self.config.get('encoding', 'utf-8')
            default_flow_style = self.config.get('default_flow_style', False)
            allow_unicode = self.config.get('allow_unicode', True)
            
            # Подготавливаем данные
            if isinstance(data, list):
                yaml_data = {"invoices": data}
            elif isinstance(data, dict):
                yaml_data = {"invoice": data}
            else:
                yaml_data = {"data": data}
            
            # Сохраняем в файл
            with open(output_path, 'w', encoding=encoding) as f:
                self.yaml.dump(
                    yaml_data,
                    f,
                    default_flow_style=default_flow_style,
                    allow_unicode=allow_unicode,
                    indent=2
                )
            
            print(f"✅ Данные успешно экспортированы в YAML: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка экспорта в YAML: {e}")
            return False
    
    def get_supported_formats(self) -> List[str]:
        """Возвращает список поддерживаемых форматов"""
        return ["yaml", "yml"] 