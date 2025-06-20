"""
Продвинутый менеджер плагинов для InvoiceGemini
Поддержка установки, обновления, удаления плагинов и управления зависимостями
"""
import os
import json
import shutil
import zipfile
import requests
import subprocess
import sys
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import hashlib
import importlib
import importlib.util
import logging

from .base_plugin import (
    BasePlugin, PluginType, PluginStatus, PluginMetadata,
    ImporterPlugin, IntegrationPlugin, WorkflowPlugin, NotificationPlugin
)
from .universal_plugin_manager import UniversalPluginManager


class PluginRepository:
    """Репозиторий плагинов"""
    
    def __init__(self, url: str, name: str = "default"):
        self.url = url
        self.name = name
        self.plugins: Dict[str, Dict] = {}
        self.last_update: Optional[datetime] = None
    
    def update_catalog(self) -> bool:
        """Обновляет каталог плагинов из репозитория"""
        try:
            response = requests.get(f"{self.url}/catalog.json", timeout=10)
            response.raise_for_status()
            
            catalog = response.json()
            self.plugins = catalog.get("plugins", {})
            self.last_update = datetime.now()
            return True
            
        except Exception as e:
            logging.error(f"Ошибка обновления каталога {self.name}: {e}")
            return False
    
    def search_plugins(self, query: str) -> List[Dict]:
        """Поиск плагинов по запросу"""
        results = []
        query_lower = query.lower()
        
        for plugin_id, plugin_info in self.plugins.items():
            if (query_lower in plugin_info.get("name", "").lower() or
                query_lower in plugin_info.get("description", "").lower() or
                query_lower in plugin_info.get("keywords", [])):
                results.append({
                    "id": plugin_id,
                    "repository": self.name,
                    **plugin_info
                })
        
        return results


class PluginInstaller:
    """Установщик плагинов"""
    
    def __init__(self, plugins_dir: str):
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.plugins_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
    
    def install_from_zip(self, zip_path: str, plugin_id: str) -> bool:
        """Устанавливает плагин из ZIP файла"""
        try:
            # Создаем временную директорию
            temp_extract_dir = self.temp_dir / f"extract_{plugin_id}"
            temp_extract_dir.mkdir(exist_ok=True)
            
            # Извлекаем ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)
            
            # Проверяем структуру плагина
            if not self._validate_plugin_structure(temp_extract_dir):
                raise ValueError("Неверная структура плагина")
            
            # Устанавливаем плагин
            plugin_dir = self.plugins_dir / plugin_id
            if plugin_dir.exists():
                shutil.rmtree(plugin_dir)
            
            shutil.move(str(temp_extract_dir), str(plugin_dir))
            
            # Устанавливаем зависимости
            self._install_dependencies(plugin_dir)
            
            return True
            
        except Exception as e:
            logging.error(f"Ошибка установки плагина {plugin_id}: {e}")
            return False
        finally:
            # Очищаем временные файлы
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
    
    def install_from_url(self, url: str, plugin_id: str) -> bool:
        """Устанавливает плагин по URL"""
        try:
            # Скачиваем файл
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Сохраняем во временный файл
            temp_file = self.temp_dir / f"{plugin_id}.zip"
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            # Устанавливаем
            result = self.install_from_zip(str(temp_file), plugin_id)
            
            # Удаляем временный файл
            temp_file.unlink(missing_ok=True)
            
            return result
            
        except Exception as e:
            logging.error(f"Ошибка установки плагина по URL {url}: {e}")
            return False
    
    def _validate_plugin_structure(self, plugin_dir: Path) -> bool:
        """Проверяет структуру плагина"""
        # Ищем файл метаданных
        metadata_file = plugin_dir / "plugin.json"
        if not metadata_file.exists():
            return False
        
        # Ищем основной файл плагина
        py_files = list(plugin_dir.glob("*_plugin.py"))
        if not py_files:
            return False
        
        return True
    
    def _install_dependencies(self, plugin_dir: Path):
        """Устанавливает зависимости плагина"""
        requirements_file = plugin_dir / "requirements.txt"
        if requirements_file.exists():
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "-r", str(requirements_file)
                ])
            except subprocess.CalledProcessError as e:
                logging.warning(f"Не удалось установить зависимости: {e}")


class AdvancedPluginManager:
    """
    Продвинутый менеджер плагинов с поддержкой установки, обновления и удаления
    """
    
    def __init__(self, plugins_dir: str = None, config_file: str = None):
        """
        Инициализация продвинутого менеджера плагинов
        
        Args:
            plugins_dir: Директория с плагинами
            config_file: Файл конфигурации
        """
        self.plugins_dir = Path(plugins_dir or "plugins/user")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = Path(config_file or "config/plugin_manager.json")
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Базовый менеджер плагинов
        self.base_manager = UniversalPluginManager(str(self.plugins_dir))
        
        # Configuration
        self.config = self._load_config()
        
        # Репозитории плагинов
        self.repositories: Dict[str, PluginRepository] = {}
        self._load_repositories()
        
        # Установщик
        self.installer = PluginInstaller(str(self.plugins_dir))
        
        # Кэш метаданных
        self.metadata_cache: Dict[str, Dict] = {}
        
        # Callbacks
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
        
        print(f"🚀 Инициализация AdvancedPluginManager...")
        print(f"📁 Директория плагинов: {self.plugins_dir}")
        print(f"📋 Репозиториев: {len(self.repositories)}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Загружает конфигурацию"""
        default_config = {
            "repositories": [
                {
                    "name": "official",
                    "url": "https://plugins.invoicegemini.com"
                }
            ],
            "auto_update": False,
            "check_signatures": True,
            "allowed_sources": ["official", "github", "local"]
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Объединяем с default_config
                    default_config.update(config)
            except Exception as e:
                logging.warning(f"Ошибка загрузки конфигурации: {e}")
        
        return default_config
    
    def _save_config(self):
        """Сохраняет конфигурацию"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Ошибка сохранения конфигурации: {e}")
    
    def _load_repositories(self):
        """Загружает репозитории плагинов"""
        for repo_config in self.config.get("repositories", []):
            repo = PluginRepository(
                url=repo_config["url"],
                name=repo_config["name"]
            )
            self.repositories[repo.name] = repo
    
    def add_repository(self, name: str, url: str) -> bool:
        """
        Добавляет новый репозиторий плагинов
        
        Args:
            name: Имя репозитория
            url: URL репозитория
            
        Returns:
            bool: True если репозиторий добавлен
        """
        try:
            repo = PluginRepository(url, name)
            if repo.update_catalog():
                self.repositories[name] = repo
                
                # Обновляем конфигурацию
                repos = self.config.get("repositories", [])
                repos.append({"name": name, "url": url})
                self.config["repositories"] = repos
                self._save_config()
                
                return True
        except Exception as e:
            logging.error(f"Ошибка добавления репозитория {name}: {e}")
        
        return False
    
    def update_repositories(self) -> Dict[str, bool]:
        """
        Обновляет каталоги всех репозиториев
        
        Returns:
            Dict[str, bool]: Результаты обновления по репозиториям
        """
        results = {}
        for name, repo in self.repositories.items():
            results[name] = repo.update_catalog()
            if self.status_callback:
                self.status_callback(f"Обновлен репозиторий {name}")
        
        return results
    
    def search_plugins(self, query: str, repository: str = None) -> List[Dict]:
        """
        Поиск плагинов по запросу
        
        Args:
            query: Поисковый запрос
            repository: Имя репозитория (если None, поиск по всем)
            
        Returns:
            List[Dict]: Список найденных плагинов
        """
        results = []
        
        repos_to_search = [self.repositories[repository]] if repository else self.repositories.values()
        
        for repo in repos_to_search:
            results.extend(repo.search_plugins(query))
        
        return results
    
    def install_plugin(self, plugin_id: str, source: str = None, **kwargs) -> bool:
        """
        Устанавливает плагин
        
        Args:
            plugin_id: Идентификатор плагина
            source: Источник установки (repository, url, file)
            **kwargs: Дополнительные параметры
            
        Returns:
            bool: True если плагин установлен
        """
        try:
            if self.progress_callback:
                self.progress_callback(0, f"Начало установки {plugin_id}")
            
            # Проверяем, не установлен ли уже плагин
            if self.is_plugin_installed(plugin_id):
                if not kwargs.get("force", False):
                    raise ValueError(f"Плагин {plugin_id} уже установлен")
                self.uninstall_plugin(plugin_id)
            
            # Определяем источник
            if source == "file" or (source is None and "file_path" in kwargs):
                result = self.installer.install_from_zip(kwargs["file_path"], plugin_id)
            elif source == "url" or (source is None and "url" in kwargs):
                result = self.installer.install_from_url(kwargs["url"], plugin_id)
            else:
                # Устанавливаем из репозитория
                result = self._install_from_repository(plugin_id, source)
            
            if result:
                if self.progress_callback:
                    self.progress_callback(100, f"Плагин {plugin_id} установлен")
                
                # Обновляем базовый менеджер
                self.base_manager._load_user_plugins()
                
                if self.status_callback:
                    self.status_callback(f"✅ Плагин {plugin_id} успешно установлен")
            
            return result
            
        except Exception as e:
            logging.error(f"Ошибка установки плагина {plugin_id}: {e}")
            if self.status_callback:
                self.status_callback(f"❌ Ошибка установки {plugin_id}: {e}")
            return False
    
    def _install_from_repository(self, plugin_id: str, repository: str = None) -> bool:
        """Устанавливает плагин из репозитория"""
        # Ищем плагин в репозиториях
        for repo_name, repo in self.repositories.items():
            if repository and repo_name != repository:
                continue
                
            if plugin_id in repo.plugins:
                plugin_info = repo.plugins[plugin_id]
                download_url = plugin_info.get("download_url")
                
                if download_url:
                    return self.installer.install_from_url(download_url, plugin_id)
        
        raise ValueError(f"Плагин {plugin_id} не найден в репозиториях")
    
    def uninstall_plugin(self, plugin_id: str) -> bool:
        """
        Удаляет плагин
        
        Args:
            plugin_id: Идентификатор плагина
            
        Returns:
            bool: True если плагин удален
        """
        try:
            plugin_dir = self.plugins_dir / plugin_id
            if plugin_dir.exists():
                # Останавливаем плагин если он активен
                self._stop_plugin(plugin_id)
                
                # Удаляем директорию
                shutil.rmtree(plugin_dir)
                
                # Обновляем базовый менеджер
                self.base_manager._load_user_plugins()
                
                if self.status_callback:
                    self.status_callback(f"✅ Плагин {plugin_id} удален")
                
                return True
            else:
                if self.status_callback:
                    self.status_callback(f"⚠️ Плагин {plugin_id} не найден")
                return False
                
        except Exception as e:
            logging.error(f"Ошибка удаления плагина {plugin_id}: {e}")
            if self.status_callback:
                self.status_callback(f"❌ Ошибка удаления {plugin_id}: {e}")
            return False
    
    def update_plugin(self, plugin_id: str) -> bool:
        """
        Обновляет плагин
        
        Args:
            plugin_id: Идентификатор плагина
            
        Returns:
            bool: True если плагин обновлен
        """
        try:
            # Получаем информацию о текущей версии
            current_info = self.get_plugin_info(plugin_id)
            if not current_info:
                return False
            
            # Ищем обновления в репозиториях
            for repo in self.repositories.values():
                if plugin_id in repo.plugins:
                    repo_info = repo.plugins[plugin_id]
                    repo_version = repo_info.get("version", "0.0.0")
                    current_version = current_info.get("version", "0.0.0")
                    
                    if self._compare_versions(repo_version, current_version) > 0:
                        # Есть обновление
                        if self.status_callback:
                            self.status_callback(f"🔄 Обновление {plugin_id} до версии {repo_version}")
                        
                        # Создаем резервную копию
                        backup_path = self._create_backup(plugin_id)
                        
                        try:
                            # Удаляем старую версию и устанавливаем новую
                            self.uninstall_plugin(plugin_id)
                            result = self._install_from_repository(plugin_id)
                            
                            if result:
                                # Удаляем резервную копию
                                if backup_path and backup_path.exists():
                                    shutil.rmtree(backup_path)
                                
                                if self.status_callback:
                                    self.status_callback(f"✅ Плагин {plugin_id} обновлен")
                                return True
                            else:
                                # Восстанавливаем из резервной копии
                                self._restore_backup(plugin_id, backup_path)
                                return False
                                
                        except Exception as e:
                            # Восстанавливаем из резервной копии
                            self._restore_backup(plugin_id, backup_path)
                            raise e
            
            if self.status_callback:
                self.status_callback(f"ℹ️ Обновлений для {plugin_id} не найдено")
            return True
            
        except Exception as e:
            logging.error(f"Ошибка обновления плагина {plugin_id}: {e}")
            if self.status_callback:
                self.status_callback(f"❌ Ошибка обновления {plugin_id}: {e}")
            return False
    
    def _create_backup(self, plugin_id: str) -> Optional[Path]:
        """Создает резервную копию плагина"""
        try:
            plugin_dir = self.plugins_dir / plugin_id
            if not plugin_dir.exists():
                return None
            
            backup_dir = self.plugins_dir / f"{plugin_id}_backup_{int(datetime.now().timestamp())}"
            shutil.copytree(plugin_dir, backup_dir)
            return backup_dir
            
        except Exception as e:
            logging.warning(f"Не удалось создать резервную копию {plugin_id}: {e}")
            return None
    
    def _restore_backup(self, plugin_id: str, backup_path: Path):
        """Восстанавливает плагин из резервной копии"""
        try:
            if backup_path and backup_path.exists():
                plugin_dir = self.plugins_dir / plugin_id
                if plugin_dir.exists():
                    shutil.rmtree(plugin_dir)
                shutil.move(str(backup_path), str(plugin_dir))
                
                if self.status_callback:
                    self.status_callback(f"🔄 Восстановлен {plugin_id} из резервной копии")
                    
        except Exception as e:
            logging.error(f"Ошибка восстановления {plugin_id}: {e}")
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """
        Сравнивает версии плагинов
        
        Returns:
            int: 1 если version1 > version2, -1 если version1 < version2, 0 если равны
        """
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        
        try:
            v1 = version_tuple(version1)
            v2 = version_tuple(version2)
            
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
            else:
                return 0
        except (ValueError, TypeError, AttributeError) as e:
            # Ошибка парсинга версий - считаем равными
            return 0
    
    def _stop_plugin(self, plugin_id: str):
        """Останавливает плагин"""
        # Удаляем экземпляры плагина из базового менеджера
        for plugin_type in PluginType:
            if plugin_id in self.base_manager.plugin_instances[plugin_type]:
                instance = self.base_manager.plugin_instances[plugin_type][plugin_id]
                try:
                    instance.cleanup()
                except (AttributeError, RuntimeError, Exception) as e:
                    # Ошибка при очистке экземпляра плагина - не критично
                    pass
                del self.base_manager.plugin_instances[plugin_type][plugin_id]
    
    def is_plugin_installed(self, plugin_id: str) -> bool:
        """Проверяет, установлен ли плагин"""
        plugin_dir = self.plugins_dir / plugin_id
        return plugin_dir.exists()
    
    def get_installed_plugins(self) -> List[Dict]:
        """Возвращает список установленных плагинов"""
        plugins = []
        
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir() and not plugin_dir.name.startswith('.'):
                info = self.get_plugin_info(plugin_dir.name)
                if info:
                    plugins.append(info)
        
        return plugins
    
    def get_plugin_info(self, plugin_id: str) -> Optional[Dict]:
        """Получает информацию о плагине"""
        plugin_dir = self.plugins_dir / plugin_id
        metadata_file = plugin_dir / "plugin.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Ошибка чтения метаданных {plugin_id}: {e}")
        
        return None
    
    def get_available_updates(self) -> List[Dict]:
        """Возвращает список доступных обновлений"""
        updates = []
        
        for plugin_id in [d.name for d in self.plugins_dir.iterdir() if d.is_dir()]:
            current_info = self.get_plugin_info(plugin_id)
            if not current_info:
                continue
            
            current_version = current_info.get("version", "0.0.0")
            
            # Ищем в репозиториях
            for repo in self.repositories.values():
                if plugin_id in repo.plugins:
                    repo_info = repo.plugins[plugin_id]
                    repo_version = repo_info.get("version", "0.0.0")
                    
                    if self._compare_versions(repo_version, current_version) > 0:
                        updates.append({
                            "plugin_id": plugin_id,
                            "current_version": current_version,
                            "available_version": repo_version,
                            "repository": repo.name
                        })
                        break
        
        return updates
    
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """Устанавливает callback для отслеживания прогресса"""
        self.progress_callback = callback
    
    def set_status_callback(self, callback: Callable[[str], None]):
        """Устанавливает callback для статусных сообщений"""
        self.status_callback = callback
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику плагинов"""
        installed = self.get_installed_plugins()
        updates = self.get_available_updates()
        
        return {
            "installed_count": len(installed),
            "available_updates": len(updates),
            "repositories": len(self.repositories),
            "plugin_types": {
                plugin_type.value: len([p for p in installed 
                                      if p.get("type") == plugin_type.value])
                for plugin_type in PluginType
            }
        } 