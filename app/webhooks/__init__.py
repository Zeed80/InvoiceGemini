"""
Модуль для обработки webhooks от различных интеграций
"""
from .paperless_webhook import (
    PaperlessWebhookServer,
    get_webhook_server,
    start_paperless_webhook
)

__all__ = [
    'PaperlessWebhookServer',
    'get_webhook_server',
    'start_paperless_webhook'
]

