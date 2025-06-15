"""
Плагин интеграции с 1C ERP системой
Синхронизация данных счетов и справочников
"""
import json
import requests
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..base_plugin import (
    IntegrationPlugin, PluginMetadata, PluginType, PluginCapability,
    PluginPriority
)


class OneCERPPlugin(IntegrationPlugin):
    """Плагин интеграции с 1C ERP"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._connection = None
        self._session = None
        self._last_sync = None
        self._sync_status = "disconnected"
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="1C ERP Integration",
            version="1.0.0",
            description="Интеграция с системой 1C ERP для синхронизации счетов и справочников",
            author="InvoiceGemini Team",
            plugin_type=PluginType.INTEGRATION,
            capabilities=[
                PluginCapability.API,
                PluginCapability.DATABASE,
                PluginCapability.REALTIME
            ],
            priority=PluginPriority.HIGH,
            config_schema={
                "required": ["server_url", "username", "password", "database"],
                "optional": {
                    "timeout": 30,
                    "sync_interval": 300,
                    "auto_sync": True,
                    "ssl_verify": True
                },
                "types": {
                    "server_url": str,
                    "username": str,
                    "password": str,
                    "database": str,
                    "timeout": int,
                    "sync_interval": int,
                    "auto_sync": bool,
                    "ssl_verify": bool
                }
            },
            dependencies=[
                "requests>=2.25.0",
                "xmltodict>=0.12.0"
            ],
            keywords=["1c", "erp", "integration", "accounting", "sync"]
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        try:
            # Проверяем конфигурацию
            required_config = ["server_url", "username", "password", "database"]
            for key in required_config:
                if key not in self.config:
                    self.set_error(f"Отсутствует обязательный параметр конфигурации: {key}")
                    return False
            
            # Создаем сессию
            self._session = requests.Session()
            self._session.timeout = self.config.get("timeout", 30)
            self._session.verify = self.config.get("ssl_verify", True)
            
            # Настраиваем аутентификацию
            auth_string = f"{self.config['username']}:{self.config['password']}"
            auth_bytes = auth_string.encode('utf-8')
            auth_b64 = base64.b64encode(auth_bytes).decode('utf-8')
            self._session.headers.update({
                'Authorization': f'Basic {auth_b64}',
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json'
            })
            
            self.status = PluginStatus.LOADED
            logging.info(f"1C ERP плагин инициализирован для {self.config['server_url']}")
            return True
            
        except Exception as e:
            self.set_error(f"Ошибка инициализации 1C ERP плагина: {e}")
            return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self._session:
            self._session.close()
            self._session = None
        self._connection = None
        self._sync_status = "disconnected"
    
    def connect(self, **kwargs) -> bool:
        """Устанавливает соединение с 1C"""
        try:
            if not self._session:
                return False
            
            # Проверяем соединение
            info_url = f"{self.config['server_url']}/hs/invoicegemini/info"
            response = self._session.get(info_url)
            
            if response.status_code == 200:
                self._connection = True
                self._sync_status = "connected"
                logging.info("Успешное подключение к 1C ERP")
                return True
            else:
                self.set_error(f"Ошибка подключения к 1C: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.set_error(f"Ошибка подключения к 1C: {e}")
            return False
    
    def disconnect(self):
        """Разрывает соединение"""
        self._connection = False
        self._sync_status = "disconnected"
        logging.info("Соединение с 1C ERP разорвано")
    
    def test_connection(self) -> bool:
        """Тестирует соединение"""
        try:
            if not self._session:
                return False
            
            test_url = f"{self.config['server_url']}/hs/invoicegemini/test"
            response = self._session.get(test_url, timeout=10)
            return response.status_code == 200
            
        except:
            return False
    
    def sync_data(self, data: Any, direction: str = "export") -> Dict[str, Any]:
        """Синхронизирует данные с 1C"""
        try:
            if not self._connection:
                if not self.connect():
                    return {"status": "error", "message": "Нет соединения с 1C"}
            
            result = {"status": "success", "synced_items": 0, "errors": []}
            
            if direction == "export":
                result = self._export_to_1c(data)
            elif direction == "import":
                result = self._import_from_1c(data)
            elif direction == "both":
                export_result = self._export_to_1c(data)
                import_result = self._import_from_1c({})
                
                result = {
                    "status": "success" if export_result["status"] == "success" and import_result["status"] == "success" else "partial",
                    "export": export_result,
                    "import": import_result
                }
            
            self._last_sync = datetime.now()
            self._sync_status = "synced"
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка синхронизации с 1C: {e}"
            self.set_error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def _export_to_1c(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Экспортирует данные в 1C"""
        try:
            # Преобразуем данные счета в формат 1C
            invoice_data = self._convert_to_1c_format(data)
            
            # Отправляем в 1C
            export_url = f"{self.config['server_url']}/hs/invoicegemini/invoice"
            response = self._session.post(export_url, json=invoice_data)
            
            if response.status_code == 200:
                response_data = response.json()
                return {
                    "status": "success",
                    "synced_items": 1,
                    "invoice_ref": response_data.get("ref"),
                    "invoice_number": response_data.get("number")
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ошибка экспорта в 1C: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {"status": "error", "message": f"Ошибка экспорта: {e}"}
    
    def _import_from_1c(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Импортирует данные из 1C"""
        try:
            # Получаем список счетов из 1C
            import_url = f"{self.config['server_url']}/hs/invoicegemini/invoices"
            
            params = {}
            if filters.get("date_from"):
                params["date_from"] = filters["date_from"]
            if filters.get("date_to"):
                params["date_to"] = filters["date_to"]
            if filters.get("organization"):
                params["organization"] = filters["organization"]
            
            response = self._session.get(import_url, params=params)
            
            if response.status_code == 200:
                invoices = response.json()
                converted_invoices = []
                
                for invoice in invoices:
                    converted = self._convert_from_1c_format(invoice)
                    converted_invoices.append(converted)
                
                return {
                    "status": "success",
                    "synced_items": len(converted_invoices),
                    "invoices": converted_invoices
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ошибка импорта из 1C: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {"status": "error", "message": f"Ошибка импорта: {e}"}
    
    def _convert_to_1c_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразует данные в формат 1C"""
        return {
            "Номер": data.get("invoice_number", ""),
            "Дата": data.get("invoice_date", ""),
            "Организация": data.get("vendor_name", ""),
            "ИНН": data.get("vendor_inn", ""),
            "КПП": data.get("vendor_kpp", ""),
            "Покупатель": data.get("customer_name", ""),
            "СуммаДокумента": data.get("total_amount", 0),
            "СуммаНДС": data.get("vat_amount", 0),
            "СуммаВключаяНДС": data.get("total_with_vat", 0),
            "Валюта": data.get("currency", "RUB"),
            "Товары": self._convert_items_to_1c(data.get("items", [])),
            "Статус": "Обработан",
            "ИсточникДанных": "InvoiceGemini"
        }
    
    def _convert_items_to_1c(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Преобразует товарные позиции в формат 1C"""
        converted_items = []
        
        for item in items:
            converted_items.append({
                "Номенклатура": item.get("name", ""),
                "Артикул": item.get("code", ""),
                "Количество": item.get("quantity", 1),
                "ЕдиницаИзмерения": item.get("unit", "шт"),
                "Цена": item.get("price", 0),
                "Сумма": item.get("amount", 0),
                "СтавкаНДС": item.get("vat_rate", 20),
                "СуммаНДС": item.get("vat_amount", 0)
            })
        
        return converted_items
    
    def _convert_from_1c_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразует данные из формата 1C"""
        return {
            "invoice_number": data.get("Номер", ""),
            "invoice_date": data.get("Дата", ""),
            "vendor_name": data.get("Организация", ""),
            "vendor_inn": data.get("ИНН", ""),
            "vendor_kpp": data.get("КПП", ""),
            "customer_name": data.get("Покупатель", ""),
            "total_amount": data.get("СуммаДокумента", 0),
            "vat_amount": data.get("СуммаНДС", 0),
            "total_with_vat": data.get("СуммаВключаяНДС", 0),
            "currency": data.get("Валюта", "RUB"),
            "items": self._convert_items_from_1c(data.get("Товары", [])),
            "status": data.get("Статус", ""),
            "source": "1C ERP"
        }
    
    def _convert_items_from_1c(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Преобразует товарные позиции из формата 1C"""
        converted_items = []
        
        for item in items:
            converted_items.append({
                "name": item.get("Номенклатура", ""),
                "code": item.get("Артикул", ""),
                "quantity": item.get("Количество", 1),
                "unit": item.get("ЕдиницаИзмерения", "шт"),
                "price": item.get("Цена", 0),
                "amount": item.get("Сумма", 0),
                "vat_rate": item.get("СтавкаНДС", 20),
                "vat_amount": item.get("СуммаНДС", 0)
            })
        
        return converted_items
    
    def get_organizations(self) -> List[Dict[str, Any]]:
        """Получает список организаций из 1C"""
        try:
            if not self._connection:
                if not self.connect():
                    return []
            
            orgs_url = f"{self.config['server_url']}/hs/invoicegemini/organizations"
            response = self._session.get(orgs_url)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.set_error(f"Ошибка получения организаций: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            self.set_error(f"Ошибка получения организаций: {e}")
            return []
    
    def get_counterparties(self) -> List[Dict[str, Any]]:
        """Получает список контрагентов из 1C"""
        try:
            if not self._connection:
                if not self.connect():
                    return []
            
            counterparties_url = f"{self.config['server_url']}/hs/invoicegemini/counterparties"
            response = self._session.get(counterparties_url)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.set_error(f"Ошибка получения контрагентов: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            self.set_error(f"Ошибка получения контрагентов: {e}")
            return []
    
    def get_nomenclature(self) -> List[Dict[str, Any]]:
        """Получает справочник номенклатуры из 1C"""
        try:
            if not self._connection:
                if not self.connect():
                    return []
            
            nomenclature_url = f"{self.config['server_url']}/hs/invoicegemini/nomenclature"
            response = self._session.get(nomenclature_url)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.set_error(f"Ошибка получения номенклатуры: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            self.set_error(f"Ошибка получения номенклатуры: {e}")
            return []
    
    def create_document(self, document_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Создает документ в 1C"""
        try:
            if not self._connection:
                if not self.connect():
                    return {"status": "error", "message": "Нет соединения с 1C"}
            
            create_url = f"{self.config['server_url']}/hs/invoicegemini/documents/{document_type}"
            response = self._session.post(create_url, json=data)
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "document": response.json()
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ошибка создания документа: HTTP {response.status_code}"
                }
                
        except Exception as e:
            return {"status": "error", "message": f"Ошибка создания документа: {e}"}
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Возвращает статус соединения"""
        return {
            "connected": self._connection or False,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "sync_status": self._sync_status,
            "server_url": self.config.get("server_url", ""),
            "database": self.config.get("database", "")
        } 