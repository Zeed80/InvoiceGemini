"""
Плагин уведомлений через Telegram
Отправка информации о обработанных счетах и статусе системы
"""
import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import asyncio
import aiohttp

from ..base_plugin import (
    NotificationPlugin, PluginMetadata, PluginType, PluginCapability,
    PluginPriority
)


class TelegramNotificationPlugin(NotificationPlugin):
    """Плагин уведомлений через Telegram"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._bot_token = None
        self._chat_ids = []
        self._session = None
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Telegram Notifications",
            version="1.0.0", 
            description="Отправка уведомлений через Telegram Bot API",
            author="InvoiceGemini Team",
            plugin_type=PluginType.NOTIFICATION,
            capabilities=[
                PluginCapability.API,
                PluginCapability.REALTIME,
                PluginCapability.ASYNC
            ],
            priority=PluginPriority.NORMAL,
            config_schema={
                "required": ["bot_token", "chat_ids"],
                "optional": {
                    "parse_mode": "HTML",
                    "disable_notification": False,
                    "timeout": 30,
                    "max_message_length": 4096
                },
                "types": {
                    "bot_token": str,
                    "chat_ids": list,
                    "parse_mode": str,
                    "disable_notification": bool,
                    "timeout": int,
                    "max_message_length": int
                }
            },
            dependencies=[
                "requests>=2.25.0",
                "aiohttp>=3.8.0"
            ],
            keywords=["telegram", "notifications", "bot", "messaging", "alerts"]
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        try:
            # Проверяем конфигурацию
            if "bot_token" not in self.config:
                self.set_error("Отсутствует bot_token в конфигурации")
                return False
            
            if "chat_ids" not in self.config or not self.config["chat_ids"]:
                self.set_error("Отсутствуют chat_ids в конфигурации")
                return False
            
            self._bot_token = self.config["bot_token"]
            self._chat_ids = self.config["chat_ids"]
            
            # Создаем сессию для синхронных запросов
            self._session = requests.Session()
            self._session.timeout = self.config.get("timeout", 30)
            
            # Проверяем токен бота
            if not self._test_bot_token():
                self.set_error("Неверный bot_token или бот недоступен")
                return False
            
            logging.info(f"Telegram плагин инициализирован для {len(self._chat_ids)} чатов")
            return True
            
        except Exception as e:
            self.set_error(f"Ошибка инициализации Telegram плагина: {e}")
            return False
    
    def cleanup(self):
        """Очистка ресурсов"""
        if self._session:
            self._session.close()
            self._session = None
    
    def _test_bot_token(self) -> bool:
        """Проверяет токен бота"""
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/getMe"
            response = self._session.get(url)
            return response.status_code == 200 and response.json().get("ok", False)
        except:
            return False
    
    def send_notification(self, message: str, **kwargs) -> bool:
        """Отправляет уведомление в Telegram"""
        try:
            chat_id = kwargs.get("chat_id")
            if chat_id:
                # Отправляем в конкретный чат
                return self._send_to_chat(chat_id, message, **kwargs)
            else:
                # Отправляем во все настроенные чаты
                success_count = 0
                for chat_id in self._chat_ids:
                    if self._send_to_chat(chat_id, message, **kwargs):
                        success_count += 1
                
                return success_count > 0
                
        except Exception as e:
            self.set_error(f"Ошибка отправки уведомления: {e}")
            return False
    
    def _send_to_chat(self, chat_id: str, message: str, **kwargs) -> bool:
        """Отправляет сообщение в конкретный чат"""
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            
            # Обрезаем сообщение если оно слишком длинное
            max_length = self.config.get("max_message_length", 4096)
            if len(message) > max_length:
                message = message[:max_length-3] + "..."
            
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": kwargs.get("parse_mode", self.config.get("parse_mode", "HTML")),
                "disable_notification": kwargs.get("disable_notification", 
                                                 self.config.get("disable_notification", False))
            }
            
            # Дополнительные параметры
            if kwargs.get("reply_markup"):
                data["reply_markup"] = kwargs["reply_markup"]
            
            if kwargs.get("disable_web_page_preview"):
                data["disable_web_page_preview"] = True
            
            response = self._session.post(url, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("ok", False)
            else:
                logging.error(f"Ошибка отправки в Telegram: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"Ошибка отправки сообщения в чат {chat_id}: {e}")
            return False
    
    async def send_notification_async(self, message: str, **kwargs) -> bool:
        """Асинхронная отправка уведомления"""
        try:
            async with aiohttp.ClientSession() as session:
                chat_id = kwargs.get("chat_id")
                if chat_id:
                    return await self._send_to_chat_async(session, chat_id, message, **kwargs)
                else:
                    tasks = []
                    for chat_id in self._chat_ids:
                        task = self._send_to_chat_async(session, chat_id, message, **kwargs)
                        tasks.append(task)
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    return any(result for result in results if isinstance(result, bool) and result)
                    
        except Exception as e:
            self.set_error(f"Ошибка асинхронной отправки: {e}")
            return False
    
    async def _send_to_chat_async(self, session: aiohttp.ClientSession, 
                                 chat_id: str, message: str, **kwargs) -> bool:
        """Асинхронная отправка в конкретный чат"""
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            
            max_length = self.config.get("max_message_length", 4096)
            if len(message) > max_length:
                message = message[:max_length-3] + "..."
            
            data = {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": kwargs.get("parse_mode", self.config.get("parse_mode", "HTML")),
                "disable_notification": kwargs.get("disable_notification", 
                                                 self.config.get("disable_notification", False))
            }
            
            async with session.post(url, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("ok", False)
                else:
                    return False
                    
        except Exception as e:
            logging.error(f"Ошибка асинхронной отправки в чат {chat_id}: {e}")
            return False
    
    def get_notification_channels(self) -> List[str]:
        """Возвращает доступные каналы уведомлений"""
        return [f"telegram_chat_{chat_id}" for chat_id in self._chat_ids]
    
    def send_invoice_notification(self, invoice_data: Dict[str, Any]) -> bool:
        """Отправляет уведомление об обработанном счете"""
        try:
            # Форматируем сообщение о счете
            message = self._format_invoice_message(invoice_data)
            
            # Добавляем inline клавиатуру с действиями
            reply_markup = {
                "inline_keyboard": [[
                    {"text": "📄 Детали", "callback_data": f"invoice_details_{invoice_data.get('id', '')}"},
                    {"text": "💾 Скачать", "callback_data": f"download_invoice_{invoice_data.get('id', '')}"}
                ]]
            }
            
            return self.send_notification(message, reply_markup=json.dumps(reply_markup))
            
        except Exception as e:
            self.set_error(f"Ошибка отправки уведомления о счете: {e}")
            return False
    
    def _format_invoice_message(self, invoice_data: Dict[str, Any]) -> str:
        """Форматирует сообщение о счете"""
        template = """
🧾 <b>Новый счет обработан</b>

📋 <b>Номер:</b> {invoice_number}
📅 <b>Дата:</b> {invoice_date}
🏢 <b>Поставщик:</b> {vendor_name}
👤 <b>Покупатель:</b> {customer_name}
💰 <b>Сумма:</b> {total_amount} {currency}
🏛️ <b>НДС:</b> {vat_amount} {currency}

⏰ <b>Обработано:</b> {processed_time}
🤖 <b>Модель:</b> {model_used}
✅ <b>Статус:</b> {status}
"""
        
        return template.format(
            invoice_number=invoice_data.get("invoice_number", "Не указан"),
            invoice_date=invoice_data.get("invoice_date", "Не указана"),
            vendor_name=invoice_data.get("vendor_name", "Не указан"),
            customer_name=invoice_data.get("customer_name", "Не указан"),
            total_amount=invoice_data.get("total_amount", 0),
            currency=invoice_data.get("currency", "RUB"),
            vat_amount=invoice_data.get("vat_amount", 0),
            processed_time=datetime.now().strftime("%H:%M:%S %d.%m.%Y"),
            model_used=invoice_data.get("model_used", "Неизвестна"),
            status=invoice_data.get("status", "Обработан")
        ).strip()
    
    def send_system_notification(self, event_type: str, details: Dict[str, Any]) -> bool:
        """Отправляет системное уведомление"""
        try:
            message = self._format_system_message(event_type, details)
            return self.send_notification(message, disable_notification=True)
            
        except Exception as e:
            self.set_error(f"Ошибка отправки системного уведомления: {e}")
            return False
    
    def _format_system_message(self, event_type: str, details: Dict[str, Any]) -> str:
        """Форматирует системное сообщение"""
        emoji_map = {
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️",
            "success": "✅",
            "startup": "🚀",
            "shutdown": "🛑",
            "update": "🔄"
        }
        
        emoji = emoji_map.get(event_type, "📢")
        
        message = f"{emoji} <b>InvoiceGemini - {event_type.upper()}</b>\n\n"
        
        if event_type == "error":
            message += f"🔴 <b>Ошибка:</b> {details.get('message', 'Неизвестная ошибка')}\n"
            if details.get('file'):
                message += f"📁 <b>Файл:</b> {details['file']}\n"
            if details.get('timestamp'):
                message += f"⏰ <b>Время:</b> {details['timestamp']}\n"
                
        elif event_type == "warning":
            message += f"🟡 <b>Предупреждение:</b> {details.get('message', '')}\n"
            
        elif event_type == "success":
            message += f"🟢 <b>Успех:</b> {details.get('message', '')}\n"
            if details.get('count'):
                message += f"📊 <b>Обработано:</b> {details['count']} документов\n"
                
        elif event_type == "startup":
            message += f"🟢 <b>Система запущена</b>\n"
            message += f"⏰ <b>Время запуска:</b> {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}\n"
            if details.get('version'):
                message += f"📦 <b>Версия:</b> {details['version']}\n"
                
        elif event_type == "shutdown":
            message += f"🔴 <b>Система остановлена</b>\n"
            message += f"⏰ <b>Время остановки:</b> {datetime.now().strftime('%H:%M:%S %d.%m.%Y')}\n"
            
        elif event_type == "update":
            message += f"🔄 <b>Обновление системы</b>\n"
            if details.get('from_version') and details.get('to_version'):
                message += f"📦 <b>Версия:</b> {details['from_version']} → {details['to_version']}\n"
        
        return message.strip()
    
    def send_statistics_report(self, stats: Dict[str, Any]) -> bool:
        """Отправляет отчет со статистикой"""
        try:
            message = self._format_statistics_message(stats)
            return self.send_notification(message)
            
        except Exception as e:
            self.set_error(f"Ошибка отправки статистики: {e}")
            return False
    
    def _format_statistics_message(self, stats: Dict[str, Any]) -> str:
        """Форматирует сообщение со статистикой"""
        template = """
📊 <b>Статистика InvoiceGemini</b>

📋 <b>Документы:</b>
• Обработано сегодня: {processed_today}
• Всего обработано: {total_processed}
• Успешно: {success_count}
• С ошибками: {error_count}

⏱️ <b>Производительность:</b>
• Среднее время обработки: {avg_processing_time}с
• Загрузка системы: {system_load}%

🤖 <b>Модели:</b>
{models_stats}

📅 <b>Период:</b> {report_period}
🕐 <b>Сгенерирован:</b> {generated_time}
"""
        
        # Форматируем статистику по моделям
        models_stats = ""
        for model, count in stats.get("models_usage", {}).items():
            models_stats += f"• {model}: {count} документов\n"
        
        return template.format(
            processed_today=stats.get("processed_today", 0),
            total_processed=stats.get("total_processed", 0),
            success_count=stats.get("success_count", 0),
            error_count=stats.get("error_count", 0),
            avg_processing_time=stats.get("avg_processing_time", 0),
            system_load=stats.get("system_load", 0),
            models_stats=models_stats.strip() or "Нет данных",
            report_period=stats.get("period", "Не указан"),
            generated_time=datetime.now().strftime("%H:%M:%S %d.%m.%Y")
        ).strip()
    
    def send_document_with_caption(self, document_path: str, caption: str, **kwargs) -> bool:
        """Отправляет документ с подписью"""
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/sendDocument"
            
            chat_id = kwargs.get("chat_id")
            if not chat_id and self._chat_ids:
                chat_id = self._chat_ids[0]  # Отправляем в первый чат по умолчанию
            
            with open(document_path, 'rb') as doc:
                files = {'document': doc}
                data = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': kwargs.get("parse_mode", self.config.get("parse_mode", "HTML"))
                }
                
                response = self._session.post(url, files=files, data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("ok", False)
                else:
                    logging.error(f"Ошибка отправки документа: HTTP {response.status_code}")
                    return False
                    
        except Exception as e:
            self.set_error(f"Ошибка отправки документа: {e}")
            return False
    
    def get_chat_info(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Получает информацию о чате"""
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/getChat"
            response = self._session.get(url, params={"chat_id": chat_id})
            
            if response.status_code == 200:
                result = response.json()
                if result.get("ok"):
                    return result.get("result")
            
            return None
            
        except Exception as e:
            logging.error(f"Ошибка получения информации о чате {chat_id}: {e}")
            return None 