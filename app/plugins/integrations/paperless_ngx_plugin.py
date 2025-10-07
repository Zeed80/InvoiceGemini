"""
Плагин интеграции с Paperless-NGX
Двусторонняя синхронизация документов, метаданных и тегов
"""
import json
import requests
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging

from ..base_plugin import (
    IntegrationPlugin, PluginMetadata, PluginType, PluginCapability,
    PluginPriority, PluginStatus
)


class PaperlessNGXPlugin(IntegrationPlugin):
    """Плагин интеграции с Paperless-NGX"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._session = None
        self._connection = None
        self._last_sync = None
        self._sync_status = "disconnected"
        self._api_version = "v2"  # Paperless-NGX API версия
        
        # Кэш для справочников
        self._correspondents_cache = {}
        self._document_types_cache = {}
        self._tags_cache = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Paperless-NGX Integration",
            version="1.0.0",
            description="Интеграция с Paperless-NGX для управления документами и автоматического тегирования",
            author="InvoiceGemini Team",
            plugin_type=PluginType.INTEGRATION,
            capabilities=[
                PluginCapability.API,
                PluginCapability.DATABASE,
                PluginCapability.REALTIME,
                PluginCapability.FILE_PROCESSING
            ],
            priority=PluginPriority.HIGH,
            config_schema={
                "required": ["server_url", "api_token"],
                "optional": {
                    "timeout": 30,
                    "sync_interval": 300,
                    "auto_sync": False,
                    "ssl_verify": True,
                    "create_correspondents": True,
                    "create_document_types": True,
                    "auto_tag": True,
                    "paperless_ai_enabled": True
                },
                "types": {
                    "server_url": str,
                    "api_token": str,
                    "timeout": int,
                    "sync_interval": int,
                    "auto_sync": bool,
                    "ssl_verify": bool,
                    "create_correspondents": bool,
                    "create_document_types": bool,
                    "auto_tag": bool,
                    "paperless_ai_enabled": bool
                }
            },
            dependencies=[
                "requests>=2.25.0"
            ],
            keywords=["paperless", "ngx", "dms", "documents", "ai", "tagging"]
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        try:
            # Проверяем конфигурацию
            if "server_url" not in self.config:
                self.set_error("Отсутствует server_url в конфигурации")
                return False
            
            if "api_token" not in self.config:
                self.set_error("Отсутствует api_token в конфигурации")
                return False
            
            # Создаем сессию
            self._session = requests.Session()
            self._session.timeout = self.config.get("timeout", 30)
            self._session.verify = self.config.get("ssl_verify", True)
            
            # Настраиваем аутентификацию
            self._session.headers.update({
                'Authorization': f'Token {self.config["api_token"]}',
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json'
            })
            
            self.status = PluginStatus.LOADED
            logging.info(f"Paperless-NGX плагин инициализирован для {self.config['server_url']}")
            return True
            
        except Exception as e:
            self.set_error(f"Ошибка инициализации Paperless-NGX плагина: {e}")
            return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self._session:
            self._session.close()
            self._session = None
        self._connection = None
        self._sync_status = "disconnected"
        self._correspondents_cache.clear()
        self._document_types_cache.clear()
        self._tags_cache.clear()
    
    def connect(self, **kwargs) -> bool:
        """Устанавливает соединение с Paperless-NGX"""
        try:
            if not self._session:
                return False
            
            # Проверяем соединение через API status endpoint
            base_url = self._get_base_url()
            status_url = f"{base_url}/api/ui_settings/"
            
            response = self._session.get(status_url)
            
            if response.status_code == 200:
                self._connection = True
                self._sync_status = "connected"
                
                # Загружаем справочники в кэш
                self._refresh_caches()
                
                logging.info("Успешное подключение к Paperless-NGX")
                return True
            else:
                self.set_error(f"Ошибка подключения к Paperless-NGX: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.set_error(f"Ошибка подключения к Paperless-NGX: {e}")
            return False
    
    def disconnect(self):
        """Разрывает соединение"""
        self._connection = False
        self._sync_status = "disconnected"
        logging.info("Соединение с Paperless-NGX разорвано")
    
    def test_connection(self) -> bool:
        """Тестирует соединение"""
        try:
            if not self._session:
                return False
            
            base_url = self._get_base_url()
            test_url = f"{base_url}/api/ui_settings/"
            response = self._session.get(test_url, timeout=10)
            return response.status_code == 200
            
        except (requests.RequestException, ValueError) as e:
            logging.warning(f"Не удалось проверить подключение к Paperless-NGX: {e}")
            return False
    
    def sync_data(self, data: Any, direction: str = "export") -> Dict[str, Any]:
        """Синхронизирует данные с Paperless-NGX"""
        try:
            if not self._connection:
                if not self.connect():
                    return {"status": "error", "message": "Нет соединения с Paperless-NGX"}
            
            result = {"status": "success", "synced_items": 0, "errors": []}
            
            if direction == "export":
                result = self._export_to_paperless(data)
            elif direction == "import":
                result = self._import_from_paperless(data)
            elif direction == "both":
                export_result = self._export_to_paperless(data)
                import_result = self._import_from_paperless({})
                
                result = {
                    "status": "success" if export_result["status"] == "success" and import_result["status"] == "success" else "partial",
                    "export": export_result,
                    "import": import_result
                }
            
            self._last_sync = datetime.now()
            self._sync_status = "synced"
            
            return result
            
        except Exception as e:
            error_msg = f"Ошибка синхронизации с Paperless-NGX: {e}"
            self.set_error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def _export_to_paperless(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Экспортирует документ в Paperless-NGX"""
        try:
            base_url = self._get_base_url()
            
            # Проверяем наличие файла
            if "file_path" not in data:
                return {"status": "error", "message": "Отсутствует путь к файлу"}
            
            file_path = Path(data["file_path"])
            if not file_path.exists():
                return {"status": "error", "message": f"Файл не найден: {file_path}"}
            
            # Подготавливаем метаданные
            metadata = self._prepare_metadata(data)
            
            # Создаем или получаем correspondent (поставщик)
            correspondent_id = None
            if data.get("sender") and self.config.get("create_correspondents", True):
                correspondent_id = self._get_or_create_correspondent(data["sender"])
            
            # Создаем или получаем document type (тип документа)
            document_type_id = None
            if data.get("category") and self.config.get("create_document_types", True):
                document_type_id = self._get_or_create_document_type(data["category"])
            
            # Получаем или создаем теги
            tag_ids = []
            if self.config.get("auto_tag", True):
                tag_ids = self._get_or_create_tags(data)
            
            # Загружаем документ
            upload_url = f"{base_url}/api/documents/post_document/"
            
            with open(file_path, 'rb') as f:
                files = {
                    'document': (file_path.name, f, 'application/pdf')
                }
                
                form_data = {
                    'title': data.get('invoice_number', file_path.stem),
                    'created': data.get('invoice_date', datetime.now().strftime('%Y-%m-%d')),
                }
                
                if correspondent_id:
                    form_data['correspondent'] = correspondent_id
                if document_type_id:
                    form_data['document_type'] = document_type_id
                if tag_ids:
                    form_data['tags'] = json.dumps(tag_ids)
                
                # Добавляем custom fields через metadata
                if metadata:
                    form_data['custom_fields'] = json.dumps(metadata)
                
                # Временно убираем Content-Type для multipart/form-data
                headers = self._session.headers.copy()
                headers.pop('Content-Type', None)
                
                response = self._session.post(
                    upload_url, 
                    files=files, 
                    data=form_data,
                    headers=headers
                )
            
            if response.status_code in [200, 201]:
                response_data = response.json()
                
                # Если включен paperless-ai, запускаем автоматическое тегирование
                if self.config.get("paperless_ai_enabled", True):
                    doc_id = response_data.get("id")
                    if doc_id:
                        self._trigger_ai_tagging(doc_id)
                
                return {
                    "status": "success",
                    "synced_items": 1,
                    "document_id": response_data.get("id"),
                    "message": f"Документ успешно загружен в Paperless-NGX (ID: {response_data.get('id')})"
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ошибка загрузки документа: HTTP {response.status_code}, {response.text}"
                }
                
        except Exception as e:
            logging.error(f"Ошибка экспорта в Paperless-NGX: {e}", exc_info=True)
            return {"status": "error", "message": f"Ошибка экспорта: {e}"}
    
    def _import_from_paperless(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Импортирует документы из Paperless-NGX"""
        try:
            base_url = self._get_base_url()
            documents_url = f"{base_url}/api/documents/"
            
            # Формируем параметры запроса
            params = {
                'page': 1,
                'page_size': filters.get('limit', 100),
                'ordering': '-created'
            }
            
            # Фильтры
            if filters.get("correspondent"):
                params['correspondent__id'] = filters["correspondent"]
            if filters.get("document_type"):
                params['document_type__id'] = filters["document_type"]
            if filters.get("tags"):
                params['tags__id__in'] = filters["tags"]
            if filters.get("created_after"):
                params['created__gte'] = filters["created_after"]
            if filters.get("created_before"):
                params['created__lte'] = filters["created_before"]
            
            response = self._session.get(documents_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("results", [])
                
                imported_docs = []
                for doc in documents:
                    converted = self._convert_from_paperless_format(doc)
                    imported_docs.append(converted)
                
                return {
                    "status": "success",
                    "synced_items": len(imported_docs),
                    "documents": imported_docs,
                    "total_count": data.get("count", len(imported_docs))
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ошибка импорта из Paperless-NGX: HTTP {response.status_code}"
                }
                
        except Exception as e:
            logging.error(f"Ошибка импорта из Paperless-NGX: {e}", exc_info=True)
            return {"status": "error", "message": f"Ошибка импорта: {e}"}
    
    def _get_or_create_correspondent(self, name: str) -> Optional[int]:
        """Получает или создает корреспондента (поставщика)"""
        try:
            # Проверяем кэш
            if name in self._correspondents_cache:
                return self._correspondents_cache[name]
            
            base_url = self._get_base_url()
            correspondents_url = f"{base_url}/api/correspondents/"
            
            # Ищем существующего
            response = self._session.get(correspondents_url, params={"name": name})
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    correspondent_id = results[0]["id"]
                    self._correspondents_cache[name] = correspondent_id
                    return correspondent_id
            
            # Создаем нового
            response = self._session.post(correspondents_url, json={"name": name})
            if response.status_code in [200, 201]:
                correspondent_id = response.json()["id"]
                self._correspondents_cache[name] = correspondent_id
                logging.info(f"Создан корреспондент в Paperless: {name} (ID: {correspondent_id})")
                return correspondent_id
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка создания корреспондента: {e}")
            return None
    
    def _get_or_create_document_type(self, name: str) -> Optional[int]:
        """Получает или создает тип документа"""
        try:
            # Проверяем кэш
            if name in self._document_types_cache:
                return self._document_types_cache[name]
            
            base_url = self._get_base_url()
            doc_types_url = f"{base_url}/api/document_types/"
            
            # Ищем существующий
            response = self._session.get(doc_types_url, params={"name": name})
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    doc_type_id = results[0]["id"]
                    self._document_types_cache[name] = doc_type_id
                    return doc_type_id
            
            # Создаем новый
            response = self._session.post(doc_types_url, json={"name": name})
            if response.status_code in [200, 201]:
                doc_type_id = response.json()["id"]
                self._document_types_cache[name] = doc_type_id
                logging.info(f"Создан тип документа в Paperless: {name} (ID: {doc_type_id})")
                return doc_type_id
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка создания типа документа: {e}")
            return None
    
    def _get_or_create_tags(self, data: Dict[str, Any]) -> List[int]:
        """Создает теги на основе данных документа"""
        try:
            tag_ids = []
            base_url = self._get_base_url()
            tags_url = f"{base_url}/api/tags/"
            
            # Создаем теги из различных полей
            tags_to_create = []
            
            # Тег валюты
            if data.get("currency"):
                tags_to_create.append(f"Валюта: {data['currency']}")
            
            # Тег НДС
            if data.get("vat_percent"):
                tags_to_create.append(f"НДС: {data['vat_percent']}%")
            
            # Тег категории (если не используется как document_type)
            if data.get("category") and not self.config.get("create_document_types", True):
                tags_to_create.append(data["category"])
            
            # Тег источника
            tags_to_create.append("InvoiceGemini")
            
            # Создаем/получаем теги
            for tag_name in tags_to_create:
                tag_id = self._get_or_create_tag(tag_name, tags_url)
                if tag_id:
                    tag_ids.append(tag_id)
            
            return tag_ids
            
        except Exception as e:
            logging.error(f"Ошибка создания тегов: {e}")
            return []
    
    def _get_or_create_tag(self, name: str, tags_url: str) -> Optional[int]:
        """Получает или создает один тег"""
        try:
            # Проверяем кэш
            if name in self._tags_cache:
                return self._tags_cache[name]
            
            # Ищем существующий
            response = self._session.get(tags_url, params={"name": name})
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    tag_id = results[0]["id"]
                    self._tags_cache[name] = tag_id
                    return tag_id
            
            # Создаем новый
            response = self._session.post(tags_url, json={"name": name})
            if response.status_code in [200, 201]:
                tag_id = response.json()["id"]
                self._tags_cache[name] = tag_id
                return tag_id
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка создания тега '{name}': {e}")
            return None
    
    def _trigger_ai_tagging(self, document_id: int):
        """Запускает AI тегирование через paperless-ai"""
        try:
            # Paperless-AI обычно работает автоматически при загрузке документа
            # Но можно вызвать принудительную обработку через задачу
            base_url = self._get_base_url()
            
            # Некоторые установки paperless-ai используют отдельный endpoint
            # Проверяем наличие и вызываем если доступен
            ai_url = f"{base_url}/api/documents/{document_id}/metadata/"
            
            response = self._session.get(ai_url)
            if response.status_code == 200:
                logging.info(f"AI тегирование инициировано для документа {document_id}")
            
        except Exception as e:
            logging.debug(f"Не удалось запустить AI тегирование: {e}")
            # Не критично, paperless-ai обычно работает автоматически
    
    def _prepare_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Подготавливает метаданные для Paperless"""
        metadata = {}
        
        # Маппинг всех полей InvoiceGemini
        field_mapping = {
            "invoice_number": "Номер счета",
            "invoice_date": "Дата счета",
            "sender": "Отправитель",
            "total": "Сумма",
            "amount_no_vat": "Сумма без НДС",
            "vat_percent": "Ставка НДС",
            "currency": "Валюта",
            "category": "Категория",
            "description": "Описание",
            "note": "Примечание"
        }
        
        for key, label in field_mapping.items():
            if key in data and data[key]:
                metadata[label] = str(data[key])
        
        return metadata
    
    def _convert_from_paperless_format(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразует документ из формата Paperless в формат InvoiceGemini"""
        try:
            # Базовая информация
            converted = {
                "paperless_id": doc.get("id"),
                "title": doc.get("title", ""),
                "created": doc.get("created"),
                "modified": doc.get("modified"),
                "download_url": doc.get("download_url"),
                "thumbnail_url": doc.get("thumbnail_url"),
                "source": "Paperless-NGX"
            }
            
            # Корреспондент -> sender
            if doc.get("correspondent_name"):
                converted["sender"] = doc["correspondent_name"]
            
            # Тип документа -> category
            if doc.get("document_type_name"):
                converted["category"] = doc["document_type_name"]
            
            # Теги
            if doc.get("tags"):
                converted["tags"] = [tag["name"] for tag in doc.get("tags", [])]
            
            # Пытаемся извлечь данные из custom fields
            if doc.get("custom_fields"):
                for field in doc["custom_fields"]:
                    if field.get("value"):
                        converted[field["field"].lower().replace(" ", "_")] = field["value"]
            
            return converted
            
        except Exception as e:
            logging.error(f"Ошибка преобразования документа из Paperless: {e}")
            return {"error": str(e), "raw_data": doc}
    
    def _refresh_caches(self):
        """Обновляет кэши справочников"""
        try:
            base_url = self._get_base_url()
            
            # Загружаем корреспондентов
            response = self._session.get(f"{base_url}/api/correspondents/", params={"page_size": 1000})
            if response.status_code == 200:
                for item in response.json().get("results", []):
                    self._correspondents_cache[item["name"]] = item["id"]
            
            # Загружаем типы документов
            response = self._session.get(f"{base_url}/api/document_types/", params={"page_size": 1000})
            if response.status_code == 200:
                for item in response.json().get("results", []):
                    self._document_types_cache[item["name"]] = item["id"]
            
            # Загружаем теги
            response = self._session.get(f"{base_url}/api/tags/", params={"page_size": 1000})
            if response.status_code == 200:
                for item in response.json().get("results", []):
                    self._tags_cache[item["name"]] = item["id"]
            
            logging.info(f"Кэши обновлены: {len(self._correspondents_cache)} корреспондентов, "
                        f"{len(self._document_types_cache)} типов документов, {len(self._tags_cache)} тегов")
            
        except Exception as e:
            logging.error(f"Ошибка обновления кэшей: {e}")
    
    def _get_base_url(self) -> str:
        """Получает базовый URL с проверкой формата"""
        url = self.config.get("server_url", "").rstrip('/')
        
        # Добавляем схему если отсутствует
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        
        return url
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Возвращает статус соединения"""
        return {
            "connected": self._connection or False,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "sync_status": self._sync_status,
            "server_url": self.config.get("server_url", ""),
            "api_version": self._api_version,
            "cached_correspondents": len(self._correspondents_cache),
            "cached_document_types": len(self._document_types_cache),
            "cached_tags": len(self._tags_cache),
            "paperless_ai_enabled": self.config.get("paperless_ai_enabled", True)
        }
    
    def get_documents(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Получает список документов из Paperless-NGX"""
        result = self._import_from_paperless(filters or {})
        if result["status"] == "success":
            return result.get("documents", [])
        return []
    
    def download_document(self, document_id: int, save_path: Path) -> bool:
        """Скачивает документ из Paperless-NGX"""
        try:
            base_url = self._get_base_url()
            download_url = f"{base_url}/api/documents/{document_id}/download/"
            
            response = self._session.get(download_url, stream=True)
            
            if response.status_code == 200:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logging.info(f"Документ {document_id} скачан: {save_path}")
                return True
            else:
                logging.error(f"Ошибка скачивания документа {document_id}: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"Ошибка скачивания документа: {e}")
            return False
    
    def update_document_tags(self, document_id: int, tags: List[str]) -> bool:
        """Обновляет теги документа"""
        try:
            base_url = self._get_base_url()
            
            # Получаем или создаем ID тегов
            tag_ids = []
            tags_url = f"{base_url}/api/tags/"
            for tag_name in tags:
                tag_id = self._get_or_create_tag(tag_name, tags_url)
                if tag_id:
                    tag_ids.append(tag_id)
            
            # Обновляем документ
            doc_url = f"{base_url}/api/documents/{document_id}/"
            response = self._session.patch(doc_url, json={"tags": tag_ids})
            
            if response.status_code == 200:
                logging.info(f"Теги документа {document_id} обновлены: {tags}")
                return True
            else:
                logging.error(f"Ошибка обновления тегов: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"Ошибка обновления тегов: {e}")
            return False

