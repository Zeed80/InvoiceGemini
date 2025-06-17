#!/usr/bin/env python3
"""
Скрипт для миграции секретов в новую систему безопасного хранения.

Использование:
    python scripts/migrate_secrets.py
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в путь Python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from typing import Dict

# Настраиваем логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Основная функция миграции секретов."""
    print("\n╔════════════════════════════════════════════════╗")
    print("║      Миграция секретов InvoiceGemini          ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    try:
        # Импортируем менеджеры
        from app.settings_manager import settings_manager
        from app.security.secrets_manager import get_secrets_manager
        
        secrets_manager = get_secrets_manager()
        
        logger.info(f"Корень проекта: {project_root}")
        logger.info(f"Директория секретов: {secrets_manager.secrets_file.parent}")
        
        # Проверяем текущий статус секретов
        print("\n📊 Текущий статус секретов:")
        status_before = {}
        for secret in secrets_manager.KNOWN_SECRETS:
            value = secrets_manager.get_secret(secret)
            is_valid = bool(value) and secrets_manager.validate_secret(secret, value)
            status_before[secret] = is_valid
            status_icon = "✅" if is_valid else "❌"
            print(f"  {status_icon} {secret}: {'Валиден' if is_valid else 'Отсутствует/Невалиден'}")
        
        # Создаем шаблон .env файла
        print("\n� Создание шаблона .env файла...")
        env_template_path = secrets_manager.export_env_template()
        print(f"   Шаблон создан: {env_template_path}")
        
        # Выполняем миграцию из старых настроек
        print("\n🔄 Миграция из старых настроек...")
        migration_results = secrets_manager.migrate_from_settings(settings_manager)
        
        for key, success in migration_results.items():
            icon = "✅" if success else "❌"
            print(f"  {icon} {key}: {'Успешно' if success else 'Не найден/невалиден'}")
        
        # Проверяем финальный статус
        print("\n� Финальный статус секретов:")
        status_after = {}
        improved_secrets = []
        for secret in secrets_manager.KNOWN_SECRETS:
            value = secrets_manager.get_secret(secret)
            is_valid = bool(value) and secrets_manager.validate_secret(secret, value)
            status_after[secret] = is_valid
            status_icon = "✅" if is_valid else "❌"
            print(f"  {status_icon} {secret}: {'Валиден' if is_valid else 'Отсутствует/Невалиден'}")
            
            # Отслеживаем улучшения
            if status_before.get(secret, False) != is_valid and is_valid:
                improved_secrets.append(secret)
        
        # Создаем резервную копию если есть секреты
        if any(status_after.values()):
            print("\n💾 Создание резервной копии...")
            backup_path = secrets_manager.create_backup()
            print(f"   Резервная копия создана: {backup_path}")
        
        # Проверяем .gitignore
        print("\n🔒 Проверка .gitignore...")
        gitignore_path = project_root / ".gitignore"
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            security_patterns = ["data/security/", ".encryption.key", ".secrets.enc", "*.enc"]
            missing_patterns = [p for p in security_patterns if p not in gitignore_content]
            
            if missing_patterns:
                print("   ⚠️  Добавьте в .gitignore следующие паттерны:")
                for pattern in missing_patterns:
                    print(f"      {pattern}")
            else:
                print("   ✅ Все паттерны безопасности присутствуют в .gitignore")
        
        # Итоговая статистика
        print("\n📈 Итоговая статистика:")
        total_secrets = len(secrets_manager.KNOWN_SECRETS)
        valid_secrets = sum(1 for v in status_after.values() if v)
        print(f"   Всего секретов: {total_secrets}")
        print(f"   Валидных секретов: {valid_secrets}")
        
        if improved_secrets:
            print(f"\n🎉 Улучшены секреты: {', '.join(improved_secrets)}")
        
        # Рекомендации
        missing_secrets = [secret for secret, valid in status_after.items() if not valid]
        if missing_secrets:
            print("\n📌 Рекомендации:")
            print(f"  📝 Необходимо настроить секреты: {', '.join(missing_secrets)}")
            print(f"  � Используйте файл {env_template_path} как образец")
            print("\n  🔑 Где получить ключи:")
            if "google_api_key" in missing_secrets:
                print(f"  🌐 Google API Key: https://console.cloud.google.com/apis/credentials")
            if "hf_token" in missing_secrets:
                print(f"  🤗 Hugging Face Token: https://huggingface.co/settings/tokens")
            if "openai_api_key" in missing_secrets:
                print(f"  🤖 OpenAI API Key: https://platform.openai.com/api-keys")
            if "anthropic_api_key" in missing_secrets:
                print(f"  🧠 Anthropic API Key: https://console.anthropic.com/account/keys")
        
        print("\n✅ Миграция завершена!")
        
    except Exception as e:
        logger.error(f"Ошибка при миграции: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 