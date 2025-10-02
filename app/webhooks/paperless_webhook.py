"""
Webhook сервер для real-time обновлений от Paperless-NGX
"""
import logging
from flask import Flask, request, jsonify
from threading import Thread
from typing import Callable, Dict, Any, Optional


class PaperlessWebhookServer:
    """Сервер для обработки webhook от Paperless-NGX"""
    
    def __init__(self, port: int = 5000, host: str = '0.0.0.0'):
        self.app = Flask(__name__)
        self.port = port
        self.host = host
        self.server_thread: Optional[Thread] = None
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Обработчики событий
        self.event_handlers: Dict[str, list] = {}
        
        # Настройка маршрутов
        self._setup_routes()
    
    def _setup_routes(self):
        """Настраивает маршруты Flask"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Проверка здоровья сервера"""
            return jsonify({"status": "ok", "service": "PaperlessWebhook"})
        
        @self.app.route('/webhook/paperless', methods=['POST'])
        def paperless_webhook():
            """Обработка webhook от Paperless"""
            try:
                data = request.json
                event_type = data.get("type", "unknown")
                
                self.logger.info(f"Получен webhook: {event_type}")
                
                # Вызов зарегистрированных обработчиков
                self._trigger_event(event_type, data)
                
                return jsonify({"status": "ok", "event": event_type})
                
            except Exception as e:
                self.logger.error(f"Ошибка обработки webhook: {e}", exc_info=True)
                return jsonify({"status": "error", "message": str(e)}), 500
    
    def register_handler(self, event_type: str, handler: Callable):
        """Регистрирует обработчик для типа события"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Зарегистрирован обработчик для {event_type}")
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Вызывает обработчики для события"""
        handlers = self.event_handlers.get(event_type, [])
        handlers += self.event_handlers.get("*", [])  # Universal handlers
        
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"Ошибка в обработчике {handler}: {e}")
    
    def start(self):
        """Запускает webhook сервер"""
        if self.running:
            self.logger.warning("Webhook сервер уже запущен")
            return
        
        def run_server():
            self.app.run(
                host=self.host,
                port=self.port,
                debug=False,
                use_reloader=False
            )
        
        self.server_thread = Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        
        self.logger.info(f"Webhook сервер запущен на {self.host}:{self.port}")
    
    def stop(self):
        """Останавливает webhook сервер"""
        # Flask не поддерживает graceful shutdown, поэтому просто отмечаем
        self.running = False
        self.logger.info("Webhook сервер остановлен")


# Глобальный экземпляр
_webhook_server = None


def get_webhook_server(port: int = 5000) -> PaperlessWebhookServer:
    """Получает глобальный экземпляр webhook сервера"""
    global _webhook_server
    if _webhook_server is None:
        _webhook_server = PaperlessWebhookServer(port=port)
    return _webhook_server


# Вспомогательные функции для быстрой настройки

def start_paperless_webhook(port: int = 5000) -> PaperlessWebhookServer:
    """Запускает webhook сервер для Paperless"""
    server = get_webhook_server(port)
    
    # Регистрируем стандартные обработчики
    def on_document_added(data):
        logging.info(f"Новый документ: {data.get('document_id')}")
    
    def on_document_updated(data):
        logging.info(f"Документ обновлен: {data.get('document_id')}")
    
    def on_document_deleted(data):
        logging.info(f"Документ удален: {data.get('document_id')}")
    
    server.register_handler("document_added", on_document_added)
    server.register_handler("document_updated", on_document_updated)
    server.register_handler("document_deleted", on_document_deleted)
    
    server.start()
    return server

