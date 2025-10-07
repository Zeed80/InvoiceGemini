"""
Плагин расширенной интеграции с Paperless-AI
Автоматическое тегирование и категоризация документов
"""
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from ..base_plugin import (
    IntegrationPlugin, PluginMetadata, PluginType, PluginCapability,
    PluginPriority, PluginStatus
)


class PaperlessAIPlugin(IntegrationPlugin):
    """Плагин интеграции с Paperless-AI для автоматического тегирования"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._session = None
        self._connection = None
        self._ai_models_cache = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Paperless-AI Integration",
            version="1.0.0",
            description="Расширенная AI интеграция для автоматического тегирования и категоризации документов в Paperless-NGX",
            author="InvoiceGemini Team",
            plugin_type=PluginType.INTEGRATION,
            capabilities=[
                PluginCapability.API,
                PluginCapability.AI,
                PluginCapability.REALTIME
            ],
            priority=PluginPriority.MEDIUM,
            config_schema={
                "required": ["server_url", "api_token"],
                "optional": {
                    "timeout": 60,
                    "auto_categorize": True,
                    "auto_tag": True,
                    "confidence_threshold": 0.7,
                    "ssl_verify": True,
                    "sync_tags_to_invoicegemini": True,
                    "use_custom_model": False,
                    "custom_model_id": None
                },
                "types": {
                    "server_url": str,
                    "api_token": str,
                    "timeout": int,
                    "auto_categorize": bool,
                    "auto_tag": bool,
                    "confidence_threshold": float,
                    "ssl_verify": bool,
                    "sync_tags_to_invoicegemini": bool,
                    "use_custom_model": bool,
                    "custom_model_id": str
                }
            },
            dependencies=[
                "requests>=2.25.0"
            ],
            keywords=["paperless", "ai", "ml", "tagging", "categorization", "automation"]
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
            self._session.timeout = self.config.get("timeout", 60)
            self._session.verify = self.config.get("ssl_verify", True)
            
            # Настраиваем аутентификацию
            self._session.headers.update({
                'Authorization': f'Token {self.config["api_token"]}',
                'Content-Type': 'application/json; charset=utf-8',
                'Accept': 'application/json'
            })
            
            self.status = PluginStatus.LOADED
            logging.info(f"Paperless-AI плагин инициализирован для {self.config['server_url']}")
            return True
            
        except Exception as e:
            self.set_error(f"Ошибка инициализации Paperless-AI плагина: {e}")
            return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self._session:
            self._session.close()
            self._session = None
        self._connection = None
        self._ai_models_cache.clear()
    
    def connect(self, **kwargs) -> bool:
        """Устанавливает соединение с Paperless-AI"""
        try:
            if not self._session:
                return False
            
            # Проверяем доступность AI функций через API
            base_url = self._get_base_url()
            
            # Пробуем получить информацию о AI моделях
            # Paperless-AI может использовать различные endpoints в зависимости от версии
            ai_endpoints = [
                "/api/ml_models/",
                "/api/ai/models/",
                "/api/classify/models/"
            ]
            
            for endpoint in ai_endpoints:
                try:
                    response = self._session.get(f"{base_url}{endpoint}")
                    if response.status_code == 200:
                        self._connection = True
                        self._ai_models_cache = response.json()
                        logging.info(f"Успешное подключение к Paperless-AI через {endpoint}")
                        return True
                except (requests.RequestException, ValueError) as e:
                    logging.debug(f"Endpoint {endpoint} недоступен: {e}")
                    continue
            
            # Если AI endpoints недоступны, проверяем базовый API
            response = self._session.get(f"{base_url}/api/ui_settings/")
            if response.status_code == 200:
                self._connection = True
                logging.info("Подключение к Paperless установлено (AI функции могут быть ограничены)")
                return True
            else:
                self.set_error(f"Ошибка подключения к Paperless-AI: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.set_error(f"Ошибка подключения к Paperless-AI: {e}")
            return False
    
    def disconnect(self):
        """Разрывает соединение"""
        self._connection = False
        logging.info("Соединение с Paperless-AI разорвано")
    
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
            logging.warning(f"Не удалось проверить подключение к Paperless-AI: {e}")
            return False
    
    def auto_tag_document(self, document_id: int) -> Dict[str, Any]:
        """Автоматически создает теги для документа используя AI"""
        try:
            if not self._connection:
                if not self.connect():
                    return {"status": "error", "message": "Нет соединения с Paperless-AI"}
            
            base_url = self._get_base_url()
            
            # Получаем текущий документ
            doc_url = f"{base_url}/api/documents/{document_id}/"
            response = self._session.get(doc_url)
            
            if response.status_code != 200:
                return {"status": "error", "message": f"Документ не найден: {document_id}"}
            
            document = response.json()
            
            # Запускаем AI анализ
            # В зависимости от версии paperless-ai, используем соответствующий endpoint
            ai_result = self._trigger_ai_analysis(document_id, document)
            
            if ai_result["status"] == "success":
                suggested_tags = ai_result.get("tags", [])
                
                # Фильтруем теги по уровню уверенности
                confidence_threshold = self.config.get("confidence_threshold", 0.7)
                filtered_tags = [
                    tag for tag in suggested_tags 
                    if tag.get("confidence", 0) >= confidence_threshold
                ]
                
                # Применяем теги если включена автоматическая установка
                if self.config.get("auto_tag", True) and filtered_tags:
                    tag_names = [tag["name"] for tag in filtered_tags]
                    self._apply_tags_to_document(document_id, tag_names)
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "suggested_tags": filtered_tags,
                    "applied_tags": filtered_tags if self.config.get("auto_tag", True) else []
                }
            else:
                return ai_result
                
        except Exception as e:
            logging.error(f"Ошибка автоматического тегирования: {e}", exc_info=True)
            return {"status": "error", "message": f"Ошибка AI тегирования: {e}"}
    
    def analyze_document_for_tags(self, invoice_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Анализирует данные документа и предлагает теги
        
        Args:
            invoice_data: Словарь с данными счета из InvoiceGemini
            
        Returns:
            Список кортежей (имя_тега, уверенность)
        """
        try:
            tags = []
            
            # Анализируем категорию документа
            if invoice_data.get('category'):
                category = invoice_data['category']
                tags.append((f"Категория: {category}", 0.9))
            
            # Анализируем данные поставщика
            if invoice_data.get('vendor'):
                vendor = invoice_data['vendor']
                tags.append((f"Поставщик: {vendor}", 0.85))
            
            # Анализируем НДС
            if invoice_data.get('tax'):
                try:
                    tax = float(str(invoice_data['tax']).replace(',', '.'))
                    if tax > 0:
                        tags.append(("С НДС", 0.95))
                    else:
                        tags.append(("Без НДС", 0.95))
                except (ValueError, TypeError) as e:
                    logging.debug(f"Не удалось распознать НДС: {e}")
            
            # Анализируем дату
            if invoice_data.get('date'):
                import re
                date_str = invoice_data['date']
                # Извлекаем год
                year_match = re.search(r'20\d{2}', date_str)
                if year_match:
                    year = year_match.group()
                    tags.append((f"Год: {year}", 0.8))
                
                # Извлекаем месяц
                month_match = re.search(r'\b(0?[1-9]|1[0-2])\b', date_str)
                if month_match:
                    month_num = int(month_match.group())
                    months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                             'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
                    if 1 <= month_num <= 12:
                        tags.append((months[month_num - 1], 0.75))
            
            # Анализируем сумму
            if invoice_data.get('total'):
                try:
                    total = float(str(invoice_data['total']).replace(',', '.').replace(' ', ''))
                    if total >= 100000:
                        tags.append(("Крупная сумма", 0.8))
                    elif total >= 10000:
                        tags.append(("Средняя сумма", 0.8))
                    else:
                        tags.append(("Малая сумма", 0.8))
                except (ValueError, TypeError) as e:
                    logging.debug(f"Не удалось распознать сумму: {e}")
            
            # Анализируем статус оплаты
            if invoice_data.get('payment_status'):
                status = invoice_data['payment_status']
                if 'оплачен' in status.lower():
                    tags.append(("Оплачено", 0.9))
                elif 'к оплате' in status.lower() or 'не оплачен' in status.lower():
                    tags.append(("К оплате", 0.9))
            
            # Фильтруем по минимальной уверенности
            confidence_threshold = self.config.get("confidence_threshold", 0.7)
            filtered_tags = [(name, conf) for name, conf in tags if conf >= confidence_threshold]
            
            return filtered_tags
            
        except Exception as e:
            logging.error(f"Ошибка анализа документа для тегов: {e}", exc_info=True)
            return []
    
    def auto_categorize_document(self, document_id: int) -> Dict[str, Any]:
        """Автоматически категоризирует документ"""
        try:
            if not self._connection:
                if not self.connect():
                    return {"status": "error", "message": "Нет соединения с Paperless-AI"}
            
            base_url = self._get_base_url()
            
            # Получаем категоризацию от AI
            categorize_url = f"{base_url}/api/documents/{document_id}/suggestions/"
            response = self._session.get(categorize_url)
            
            if response.status_code == 200:
                suggestions = response.json()
                
                # Извлекаем предложения по категориям
                category_suggestions = {
                    "correspondent": suggestions.get("correspondents", []),
                    "document_type": suggestions.get("document_types", []),
                    "storage_path": suggestions.get("storage_paths", [])
                }
                
                # Применяем категории если включена автоматическая категоризация
                if self.config.get("auto_categorize", True):
                    self._apply_categorization(document_id, category_suggestions)
                
                return {
                    "status": "success",
                    "document_id": document_id,
                    "suggestions": category_suggestions
                }
            else:
                return {
                    "status": "error",
                    "message": f"Ошибка получения предложений: HTTP {response.status_code}"
                }
                
        except Exception as e:
            logging.error(f"Ошибка автоматической категоризации: {e}", exc_info=True)
            return {"status": "error", "message": f"Ошибка AI категоризации: {e}"}
    
    def sync_tags_from_paperless(self, document_id: int, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Синхронизирует теги из Paperless в InvoiceGemini"""
        try:
            if not self.config.get("sync_tags_to_invoicegemini", True):
                return {"status": "skipped", "message": "Синхронизация тегов отключена"}
            
            base_url = self._get_base_url()
            doc_url = f"{base_url}/api/documents/{document_id}/"
            
            response = self._session.get(doc_url)
            if response.status_code == 200:
                document = response.json()
                paperless_tags = [tag["name"] for tag in document.get("tags", [])]
                
                # Объединяем с существующими тегами в invoice_data
                existing_tags = invoice_data.get("tags", [])
                if isinstance(existing_tags, str):
                    existing_tags = [existing_tags]
                
                combined_tags = list(set(existing_tags + paperless_tags))
                
                return {
                    "status": "success",
                    "tags": combined_tags,
                    "paperless_tags": paperless_tags,
                    "original_tags": existing_tags
                }
            else:
                return {"status": "error", "message": f"Ошибка получения документа: HTTP {response.status_code}"}
                
        except Exception as e:
            logging.error(f"Ошибка синхронизации тегов: {e}")
            return {"status": "error", "message": f"Ошибка синхронизации: {e}"}
    
    def _trigger_ai_analysis(self, document_id: int, document: Dict[str, Any]) -> Dict[str, Any]:
        """Запускает AI анализ документа"""
        try:
            base_url = self._get_base_url()
            
            # Пробуем различные методы AI анализа
            
            # Метод 1: Через задачу классификации
            classify_url = f"{base_url}/api/tasks/"
            task_data = {
                "type": "classify_document",
                "document_id": document_id
            }
            
            response = self._session.post(classify_url, json=task_data)
            if response.status_code in [200, 201, 202]:
                # Задача создана, ждем результат
                task_id = response.json().get("task_id")
                if task_id:
                    return self._wait_for_task_completion(task_id)
            
            # Метод 2: Через прямой анализ текста (если доступен content)
            if document.get("content"):
                tags = self._analyze_content_for_tags(document["content"])
                return {
                    "status": "success",
                    "tags": tags
                }
            
            # Метод 3: Через suggestions endpoint
            suggestions_url = f"{base_url}/api/documents/{document_id}/suggestions/"
            response = self._session.get(suggestions_url)
            if response.status_code == 200:
                suggestions = response.json()
                
                # Извлекаем теги из предложений
                tags = []
                for tag in suggestions.get("tags", []):
                    tags.append({
                        "name": tag.get("name", ""),
                        "confidence": tag.get("score", 0.5)
                    })
                
                return {
                    "status": "success",
                    "tags": tags
                }
            
            # Fallback: возвращаем пустой результат
            return {
                "status": "success",
                "tags": [],
                "message": "AI анализ недоступен, используйте ручное тегирование"
            }
            
        except Exception as e:
            logging.error(f"Ошибка AI анализа: {e}")
            return {"status": "error", "message": f"Ошибка AI анализа: {e}"}
    
    def _analyze_content_for_tags(self, content: str) -> List[Dict[str, Any]]:
        """Анализирует текст документа для создания тегов"""
        tags = []
        
        # Простой анализ ключевых слов (можно расширить с помощью NLP)
        keywords_mapping = {
            "счет": {"name": "Счет", "confidence": 0.9},
            "фактура": {"name": "Фактура", "confidence": 0.9},
            "накладная": {"name": "Накладная", "confidence": 0.9},
            "акт": {"name": "Акт", "confidence": 0.85},
            "договор": {"name": "Договор", "confidence": 0.85},
            "ндс": {"name": "С НДС", "confidence": 0.8},
            "без ндс": {"name": "Без НДС", "confidence": 0.8},
            "оплачен": {"name": "Оплачено", "confidence": 0.75},
            "к оплате": {"name": "К оплате", "confidence": 0.75}
        }
        
        content_lower = content.lower()
        for keyword, tag_info in keywords_mapping.items():
            if keyword in content_lower:
                tags.append(tag_info)
        
        return tags
    
    def _wait_for_task_completion(self, task_id: str, max_wait: int = 30) -> Dict[str, Any]:
        """Ожидает завершения задачи AI"""
        import time
        
        base_url = self._get_base_url()
        task_url = f"{base_url}/api/tasks/{task_id}/"
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = self._session.get(task_url)
                if response.status_code == 200:
                    task = response.json()
                    
                    if task.get("status") == "SUCCESS":
                        return {
                            "status": "success",
                            "tags": task.get("result", {}).get("tags", [])
                        }
                    elif task.get("status") == "FAILURE":
                        return {
                            "status": "error",
                            "message": task.get("error", "Ошибка выполнения задачи")
                        }
                    
                    # Задача еще выполняется
                    time.sleep(2)
                else:
                    break
                    
            except Exception as e:
                logging.error(f"Ошибка проверки статуса задачи: {e}")
                break
        
        return {
            "status": "timeout",
            "message": "Превышено время ожидания AI анализа"
        }
    
    def _apply_tags_to_document(self, document_id: int, tag_names: List[str]) -> bool:
        """Применяет теги к документу"""
        try:
            base_url = self._get_base_url()
            
            # Получаем или создаем ID тегов
            tag_ids = []
            tags_url = f"{base_url}/api/tags/"
            
            for tag_name in tag_names:
                # Ищем существующий тег
                response = self._session.get(tags_url, params={"name": tag_name})
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    if results:
                        tag_ids.append(results[0]["id"])
                        continue
                
                # Создаем новый тег
                response = self._session.post(tags_url, json={"name": tag_name})
                if response.status_code in [200, 201]:
                    tag_ids.append(response.json()["id"])
            
            # Обновляем документ
            doc_url = f"{base_url}/api/documents/{document_id}/"
            
            # Получаем текущие теги документа
            response = self._session.get(doc_url)
            if response.status_code == 200:
                current_tags = [tag["id"] for tag in response.json().get("tags", [])]
                
                # Объединяем с новыми тегами
                all_tags = list(set(current_tags + tag_ids))
                
                # Обновляем
                response = self._session.patch(doc_url, json={"tags": all_tags})
                
                if response.status_code == 200:
                    logging.info(f"Теги применены к документу {document_id}: {tag_names}")
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Ошибка применения тегов: {e}")
            return False
    
    def _apply_categorization(self, document_id: int, suggestions: Dict[str, List]) -> bool:
        """Применяет категоризацию к документу"""
        try:
            base_url = self._get_base_url()
            doc_url = f"{base_url}/api/documents/{document_id}/"
            
            update_data = {}
            
            # Применяем correspondent (выбираем с наивысшим score)
            if suggestions.get("correspondent"):
                best_correspondent = max(suggestions["correspondent"], key=lambda x: x.get("score", 0))
                if best_correspondent.get("score", 0) >= self.config.get("confidence_threshold", 0.7):
                    update_data["correspondent"] = best_correspondent["id"]
            
            # Применяем document_type
            if suggestions.get("document_type"):
                best_doc_type = max(suggestions["document_type"], key=lambda x: x.get("score", 0))
                if best_doc_type.get("score", 0) >= self.config.get("confidence_threshold", 0.7):
                    update_data["document_type"] = best_doc_type["id"]
            
            # Применяем storage_path
            if suggestions.get("storage_path"):
                best_storage = max(suggestions["storage_path"], key=lambda x: x.get("score", 0))
                if best_storage.get("score", 0) >= self.config.get("confidence_threshold", 0.7):
                    update_data["storage_path"] = best_storage["id"]
            
            if update_data:
                response = self._session.patch(doc_url, json=update_data)
                if response.status_code == 200:
                    logging.info(f"Категоризация применена к документу {document_id}")
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"Ошибка применения категоризации: {e}")
            return False
    
    def _get_base_url(self) -> str:
        """Получает базовый URL с проверкой формата"""
        url = self.config.get("server_url", "").rstrip('/')
        
        # Добавляем схему если отсутствует
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        
        return url
    
    def get_ai_models_info(self) -> Dict[str, Any]:
        """Возвращает информацию о доступных AI моделях"""
        return {
            "available_models": self._ai_models_cache,
            "current_model": self.config.get("custom_model_id"),
            "auto_categorize": self.config.get("auto_categorize", True),
            "auto_tag": self.config.get("auto_tag", True),
            "confidence_threshold": self.config.get("confidence_threshold", 0.7)
        }

