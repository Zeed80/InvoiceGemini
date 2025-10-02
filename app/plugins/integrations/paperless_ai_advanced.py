"""
Продвинутый плагин интеграции с Paperless-AI
Расширенное AI тегирование с обучением и аналитикой
"""
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import defaultdict

from ..base_plugin import (
    IntegrationPlugin, PluginMetadata, PluginType, PluginCapability,
    PluginPriority, PluginStatus
)


class AITaggingStatistics:
    """Статистика AI тегирования"""
    
    def __init__(self):
        self.total_documents = 0
        self.total_tags_suggested = 0
        self.total_tags_applied = 0
        self.total_tags_rejected = 0
        self.tag_accuracy = {}  # tag_name -> (applied, rejected)
        self.model_performance = {}  # model_id -> performance metrics
        self.session_start = datetime.now()
    
    def record_suggestion(self, tag: str, confidence: float, applied: bool):
        """Записывает предложение тега"""
        self.total_tags_suggested += 1
        
        if applied:
            self.total_tags_applied += 1
        else:
            self.total_tags_rejected += 1
        
        if tag not in self.tag_accuracy:
            self.tag_accuracy[tag] = {"applied": 0, "rejected": 0, "avg_confidence": []}
        
        if applied:
            self.tag_accuracy[tag]["applied"] += 1
        else:
            self.tag_accuracy[tag]["rejected"] += 1
        
        self.tag_accuracy[tag]["avg_confidence"].append(confidence)
    
    def get_tag_accuracy(self, tag: str) -> float:
        """Возвращает точность тега"""
        if tag not in self.tag_accuracy:
            return 0.0
        
        stats = self.tag_accuracy[tag]
        total = stats["applied"] + stats["rejected"]
        
        if total == 0:
            return 0.0
        
        return stats["applied"] / total
    
    def get_top_tags(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Возвращает топ тегов по точности"""
        tag_scores = []
        
        for tag, stats in self.tag_accuracy.items():
            accuracy = self.get_tag_accuracy(tag)
            total_uses = stats["applied"] + stats["rejected"]
            
            # Взвешенная оценка (точность * использование)
            score = accuracy * total_uses
            tag_scores.append((tag, score))
        
        tag_scores.sort(key=lambda x: x[1], reverse=True)
        return tag_scores[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Экспортирует статистику в словарь"""
        return {
            "total_documents": self.total_documents,
            "total_tags_suggested": self.total_tags_suggested,
            "total_tags_applied": self.total_tags_applied,
            "total_tags_rejected": self.total_tags_rejected,
            "acceptance_rate": (
                self.total_tags_applied / self.total_tags_suggested 
                if self.total_tags_suggested > 0 else 0
            ),
            "tag_accuracy": {
                tag: {
                    "accuracy": self.get_tag_accuracy(tag),
                    "applied": stats["applied"],
                    "rejected": stats["rejected"],
                    "avg_confidence": (
                        sum(stats["avg_confidence"]) / len(stats["avg_confidence"])
                        if stats["avg_confidence"] else 0
                    )
                }
                for tag, stats in self.tag_accuracy.items()
            },
            "session_duration": str(datetime.now() - self.session_start),
            "top_tags": self.get_top_tags()
        }


class CustomTaggingRule:
    """Кастомное правило тегирования"""
    
    def __init__(self, rule_id: str, name: str, pattern: str, tags: List[str], 
                 enabled: bool = True, confidence: float = 1.0):
        self.rule_id = rule_id
        self.name = name
        self.pattern = pattern  # Regex или ключевые слова
        self.tags = tags
        self.enabled = enabled
        self.confidence = confidence
        self.matches = 0
    
    def matches_content(self, content: str) -> bool:
        """Проверяет совпадение с контентом"""
        import re
        
        try:
            # Пробуем regex
            if re.search(self.pattern, content, re.IGNORECASE):
                self.matches += 1
                return True
        except re.error:
            # Если не regex, ищем как обычную строку
            if self.pattern.lower() in content.lower():
                self.matches += 1
                return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Экспортирует правило в словарь"""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "pattern": self.pattern,
            "tags": self.tags,
            "enabled": self.enabled,
            "confidence": self.confidence,
            "matches": self.matches
        }


class PaperlessAIAdvanced(IntegrationPlugin):
    """Продвинутый плагин Paperless-AI с расширенными возможностями"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._session = None
        self._connection = None
        self._ai_models_cache = {}
        
        # Статистика и аналитика
        self.statistics = AITaggingStatistics()
        
        # Кастомные правила тегирования
        self.custom_rules: Dict[str, CustomTaggingRule] = {}
        
        # История предложений для обучения
        self.suggestion_history = []
        
        # Кэш тегов для быстрого доступа
        self._tags_cache = {}
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Paperless-AI Advanced",
            version="2.0.0",
            description="Продвинутая AI интеграция с обучением, статистикой и кастомными правилами тегирования",
            author="InvoiceGemini Team",
            plugin_type=PluginType.INTEGRATION,
            capabilities=[
                PluginCapability.API,
                PluginCapability.AI,
                PluginCapability.REALTIME,
                PluginCapability.TRAINING
            ],
            priority=PluginPriority.HIGH,
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
                    "custom_model_id": None,
                    "enable_learning": True,
                    "enable_custom_rules": True,
                    "statistics_enabled": True,
                    "min_tag_confidence": 0.5,
                    "max_tags_per_document": 10
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
                    "custom_model_id": str,
                    "enable_learning": bool,
                    "enable_custom_rules": bool,
                    "statistics_enabled": bool,
                    "min_tag_confidence": float,
                    "max_tags_per_document": int
                }
            },
            dependencies=[
                "requests>=2.25.0"
            ],
            keywords=["paperless", "ai", "ml", "tagging", "learning", "analytics", "automation"]
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        try:
            # Базовая инициализация
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
            
            # Загружаем кастомные правила
            if self.config.get("enable_custom_rules", True):
                self._load_custom_rules()
            
            # Загружаем статистику
            if self.config.get("statistics_enabled", True):
                self._load_statistics()
            
            self.status = PluginStatus.LOADED
            logging.info(f"Paperless-AI Advanced плагин инициализирован для {self.config['server_url']}")
            return True
            
        except Exception as e:
            self.set_error(f"Ошибка инициализации Paperless-AI Advanced: {e}")
            return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        # Сохраняем статистику перед закрытием
        if self.config.get("statistics_enabled", True):
            self._save_statistics()
        
        # Сохраняем правила
        if self.config.get("enable_custom_rules", True):
            self._save_custom_rules()
        
        if self._session:
            self._session.close()
            self._session = None
        
        self._connection = None
        self._ai_models_cache.clear()
        self._tags_cache.clear()
    
    def connect(self, **kwargs) -> bool:
        """Устанавливает соединение с Paperless-AI"""
        try:
            if not self._session:
                return False
            
            base_url = self._get_base_url()
            
            # Проверяем доступность AI функций
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
                except:
                    continue
            
            # Fallback на базовый API
            response = self._session.get(f"{base_url}/api/ui_settings/")
            if response.status_code == 200:
                self._connection = True
                logging.info("Подключение к Paperless установлено (AI функции могут быть ограничены)")
                return True
            
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
            
        except:
            return False
    
    def smart_tag_document(self, document_id: int, content: str = None) -> Dict[str, Any]:
        """
        Умное тегирование документа с использованием AI + кастомных правил
        
        Args:
            document_id: ID документа в Paperless
            content: Текст документа (опционально, будет получен автоматически)
        """
        try:
            if not self._connection:
                if not self.connect():
                    return {"status": "error", "message": "Нет соединения с Paperless-AI"}
            
            base_url = self._get_base_url()
            
            # Получаем документ если нужен content
            if content is None:
                doc_url = f"{base_url}/api/documents/{document_id}/"
                response = self._session.get(doc_url)
                
                if response.status_code != 200:
                    return {"status": "error", "message": f"Документ не найден: {document_id}"}
                
                document = response.json()
                content = document.get("content", "")
            
            all_suggested_tags = []
            
            # 1. AI тегирование
            ai_tags = self._get_ai_tags(document_id, content)
            all_suggested_tags.extend(ai_tags)
            
            # 2. Кастомные правила
            if self.config.get("enable_custom_rules", True):
                custom_tags = self._apply_custom_rules(content)
                all_suggested_tags.extend(custom_tags)
            
            # 3. Фильтруем по уверенности и лимиту
            filtered_tags = self._filter_and_rank_tags(all_suggested_tags)
            
            # 4. Применяем теги если включено
            applied_tags = []
            if self.config.get("auto_tag", True):
                applied_tags = self._apply_tags_to_document(
                    document_id, 
                    [tag["name"] for tag in filtered_tags]
                )
            
            # 5. Записываем статистику
            if self.config.get("statistics_enabled", True):
                self.statistics.total_documents += 1
                for tag in filtered_tags:
                    was_applied = tag["name"] in applied_tags
                    self.statistics.record_suggestion(
                        tag["name"], 
                        tag["confidence"], 
                        was_applied
                    )
            
            # 6. Сохраняем для обучения
            if self.config.get("enable_learning", True):
                self.suggestion_history.append({
                    "document_id": document_id,
                    "timestamp": datetime.now().isoformat(),
                    "suggested_tags": filtered_tags,
                    "applied_tags": applied_tags
                })
            
            return {
                "status": "success",
                "document_id": document_id,
                "suggested_tags": filtered_tags,
                "applied_tags": applied_tags if self.config.get("auto_tag") else [],
                "ai_tags_count": len(ai_tags),
                "custom_rules_count": len(custom_tags) if self.config.get("enable_custom_rules") else 0
            }
            
        except Exception as e:
            logging.error(f"Ошибка умного тегирования: {e}", exc_info=True)
            return {"status": "error", "message": f"Ошибка умного тегирования: {e}"}
    
    def _get_ai_tags(self, document_id: int, content: str) -> List[Dict[str, Any]]:
        """Получает AI теги от Paperless"""
        try:
            base_url = self._get_base_url()
            suggestions_url = f"{base_url}/api/documents/{document_id}/suggestions/"
            
            response = self._session.get(suggestions_url)
            if response.status_code == 200:
                suggestions = response.json()
                
                tags = []
                for tag in suggestions.get("tags", []):
                    tags.append({
                        "name": tag.get("name", ""),
                        "confidence": tag.get("score", 0.5),
                        "source": "ai",
                        "model": self.config.get("custom_model_id", "default")
                    })
                
                return tags
            
            # Fallback: анализ контента
            return self._analyze_content_for_tags(content)
            
        except Exception as e:
            logging.error(f"Ошибка получения AI тегов: {e}")
            return []
    
    def _apply_custom_rules(self, content: str) -> List[Dict[str, Any]]:
        """Применяет кастомные правила тегирования"""
        tags = []
        
        for rule_id, rule in self.custom_rules.items():
            if not rule.enabled:
                continue
            
            if rule.matches_content(content):
                for tag_name in rule.tags:
                    tags.append({
                        "name": tag_name,
                        "confidence": rule.confidence,
                        "source": "custom_rule",
                        "rule_id": rule_id,
                        "rule_name": rule.name
                    })
        
        return tags
    
    def _analyze_content_for_tags(self, content: str) -> List[Dict[str, Any]]:
        """Анализирует контент для создания тегов (fallback)"""
        tags = []
        
        keywords_mapping = {
            "счет": {"name": "Счет", "confidence": 0.9},
            "фактура": {"name": "Фактура", "confidence": 0.9},
            "накладная": {"name": "Накладная", "confidence": 0.9},
            "акт": {"name": "Акт", "confidence": 0.85},
            "договор": {"name": "Договор", "confidence": 0.85},
            "ндс": {"name": "С НДС", "confidence": 0.8},
            "без ндс": {"name": "Без НДС", "confidence": 0.8},
            "оплачен": {"name": "Оплачено", "confidence": 0.75},
            "к оплате": {"name": "К оплате", "confidence": 0.75},
            "услуг": {"name": "Услуги", "confidence": 0.7},
            "товар": {"name": "Товары", "confidence": 0.7}
        }
        
        content_lower = content.lower()
        for keyword, tag_info in keywords_mapping.items():
            if keyword in content_lower:
                tags.append({
                    "name": tag_info["name"],
                    "confidence": tag_info["confidence"],
                    "source": "keyword_analysis"
                })
        
        return tags
    
    def _filter_and_rank_tags(self, tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Фильтрует и ранжирует теги"""
        min_confidence = self.config.get("min_tag_confidence", 0.5)
        max_tags = self.config.get("max_tags_per_document", 10)
        
        # Фильтруем по минимальной уверенности
        filtered = [tag for tag in tags if tag["confidence"] >= min_confidence]
        
        # Удаляем дубликаты, оставляя тег с большей уверенностью
        unique_tags = {}
        for tag in filtered:
            tag_name = tag["name"]
            if tag_name not in unique_tags or tag["confidence"] > unique_tags[tag_name]["confidence"]:
                unique_tags[tag_name] = tag
        
        # Сортируем по уверенности
        sorted_tags = sorted(unique_tags.values(), key=lambda x: x["confidence"], reverse=True)
        
        # Ограничиваем количество
        return sorted_tags[:max_tags]
    
    def _apply_tags_to_document(self, document_id: int, tag_names: List[str]) -> List[str]:
        """Применяет теги к документу"""
        try:
            base_url = self._get_base_url()
            
            # Получаем или создаем ID тегов
            tag_ids = []
            tags_url = f"{base_url}/api/tags/"
            
            for tag_name in tag_names:
                tag_id = self._get_or_create_tag(tag_name, tags_url)
                if tag_id:
                    tag_ids.append(tag_id)
            
            # Обновляем документ
            doc_url = f"{base_url}/api/documents/{document_id}/"
            
            # Получаем текущие теги
            response = self._session.get(doc_url)
            if response.status_code == 200:
                current_tags = [tag["id"] for tag in response.json().get("tags", [])]
                
                # Объединяем с новыми тегами
                all_tags = list(set(current_tags + tag_ids))
                
                # Обновляем
                response = self._session.patch(doc_url, json={"tags": all_tags})
                
                if response.status_code == 200:
                    logging.info(f"Теги применены к документу {document_id}: {tag_names}")
                    return tag_names
            
            return []
            
        except Exception as e:
            logging.error(f"Ошибка применения тегов: {e}")
            return []
    
    def _get_or_create_tag(self, name: str, tags_url: str) -> Optional[int]:
        """Получает или создает тег"""
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
    
    def add_custom_rule(self, rule: CustomTaggingRule) -> bool:
        """Добавляет кастомное правило"""
        try:
            self.custom_rules[rule.rule_id] = rule
            self._save_custom_rules()
            logging.info(f"Добавлено правило: {rule.name}")
            return True
        except Exception as e:
            logging.error(f"Ошибка добавления правила: {e}")
            return False
    
    def remove_custom_rule(self, rule_id: str) -> bool:
        """Удаляет кастомное правило"""
        try:
            if rule_id in self.custom_rules:
                del self.custom_rules[rule_id]
                self._save_custom_rules()
                logging.info(f"Удалено правило: {rule_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Ошибка удаления правила: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику AI тегирования"""
        return self.statistics.to_dict()
    
    def export_learning_data(self, file_path: Path) -> bool:
        """Экспортирует данные для обучения"""
        try:
            learning_data = {
                "suggestion_history": self.suggestion_history,
                "statistics": self.statistics.to_dict(),
                "custom_rules": {
                    rule_id: rule.to_dict() 
                    for rule_id, rule in self.custom_rules.items()
                },
                "exported_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(learning_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Данные для обучения экспортированы в {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Ошибка экспорта данных: {e}")
            return False
    
    def _load_custom_rules(self):
        """Загружает кастомные правила"""
        try:
            rules_file = Path("data/paperless_custom_rules.json")
            if rules_file.exists():
                with open(rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                
                for rule_id, rule_dict in rules_data.items():
                    rule = CustomTaggingRule(
                        rule_id=rule_dict["rule_id"],
                        name=rule_dict["name"],
                        pattern=rule_dict["pattern"],
                        tags=rule_dict["tags"],
                        enabled=rule_dict.get("enabled", True),
                        confidence=rule_dict.get("confidence", 1.0)
                    )
                    rule.matches = rule_dict.get("matches", 0)
                    self.custom_rules[rule_id] = rule
                
                logging.info(f"Загружено {len(self.custom_rules)} кастомных правил")
        except Exception as e:
            logging.error(f"Ошибка загрузки правил: {e}")
    
    def _save_custom_rules(self):
        """Сохраняет кастомные правила"""
        try:
            rules_file = Path("data/paperless_custom_rules.json")
            rules_file.parent.mkdir(parents=True, exist_ok=True)
            
            rules_data = {
                rule_id: rule.to_dict() 
                for rule_id, rule in self.custom_rules.items()
            }
            
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            logging.debug(f"Сохранено {len(self.custom_rules)} правил")
        except Exception as e:
            logging.error(f"Ошибка сохранения правил: {e}")
    
    def _load_statistics(self):
        """Загружает статистику"""
        try:
            stats_file = Path("data/paperless_ai_statistics.json")
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats_data = json.load(f)
                
                self.statistics.total_documents = stats_data.get("total_documents", 0)
                self.statistics.total_tags_suggested = stats_data.get("total_tags_suggested", 0)
                self.statistics.total_tags_applied = stats_data.get("total_tags_applied", 0)
                self.statistics.total_tags_rejected = stats_data.get("total_tags_rejected", 0)
                
                logging.info("Статистика загружена")
        except Exception as e:
            logging.error(f"Ошибка загрузки статистики: {e}")
    
    def _save_statistics(self):
        """Сохраняет статистику"""
        try:
            stats_file = Path("data/paperless_ai_statistics.json")
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.statistics.to_dict(), f, indent=2, ensure_ascii=False)
            
            logging.debug("Статистика сохранена")
        except Exception as e:
            logging.error(f"Ошибка сохранения статистики: {e}")
    
    def _get_base_url(self) -> str:
        """Получает базовый URL"""
        url = self.config.get("server_url", "").rstrip('/')
        
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        
        return url

