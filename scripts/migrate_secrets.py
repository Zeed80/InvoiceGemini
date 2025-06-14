#!/usr/bin/env python3
"""
Скрипт для автоматической миграции секретов InvoiceGemini в безопасную систему.
Выполняет миграцию из старых незащищенных файлов в зашифрованное хранилище.

Использование:
    python scripts/migrate_secrets.py
"""

import os
import sys
import logging
from pathlib import Path

# Добавляем корень проекта в sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Настройка логирования для скрипта миграции."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Основная функция миграции."""
    logger = setup_logging()
    
    print("🔒 InvoiceGemini Security Migration Tool")
    print("=" * 50)
    
    try:
        # Импортируем менеджер секретов
        from config.secrets import SecretsManager
        
        # Создаем экземпляр менеджера
        secrets_manager = SecretsManager(project_root=str(project_root))
        
        logger.info(f"Корень проекта: {secrets_manager.project_root}")
        logger.info(f"Директория секретов: {secrets_manager.secrets_dir}")
        
        # Проверяем текущий статус секретов
        logger.info("Проверка текущего статуса секретов...")
        status_before = secrets_manager.get_all_secret_status()
        
        print("\n📊 Статус секретов ДО миграции:")
        for secret, is_valid in status_before.items():
            status_icon = "✅" if is_valid else "❌"
            print(f"  {status_icon} {secret}: {'Валиден' if is_valid else 'Отсутствует/Невалиден'}")
        
        # Создаем .env файл из шаблона если его нет
        print("\n📁 Проверка .env файла...")
        if secrets_manager.create_env_template():
            print("  ✅ Создан .env файл на основе env.example")
            print("  ⚠️  ВНИМАНИЕ: Заполните .env файл реальными API ключами!")
        else:
            print("  ℹ️  Файл .env уже существует")
        
        # Выполняем миграцию старых секретов
        print("\n🔄 Миграция старых секретов...")
        secrets_manager.cleanup_legacy_secrets()
        
        # Проверяем статус после миграции
        print("\n🔍 Проверка статуса после миграции...")
        status_after = secrets_manager.get_all_secret_status()
        
        print("\n📊 Статус секретов ПОСЛЕ миграции:")
        improved_secrets = []
        for secret, is_valid in status_after.items():
            status_icon = "✅" if is_valid else "❌"
            print(f"  {status_icon} {secret}: {'Валиден' if is_valid else 'Отсутствует/Невалиден'}")
            
            # Проверяем улучшения
            if status_before.get(secret, False) != is_valid and is_valid:
                improved_secrets.append(secret)
        
        # Отчет об улучшениях
        if improved_secrets:
            print(f"\n🎉 Улучшены секреты: {', '.join(improved_secrets)}")
        
        # Финальная сводка
        valid_count = sum(status_after.values())
        total_count = len(status_after)
        
        print(f"\n📈 Итоговая статистика:")
        print(f"  Валидных секретов: {valid_count}/{total_count}")
        print(f"  Процент готовности: {(valid_count/total_count)*100:.1f}%")
        
        # Рекомендации
        print(f"\n💡 Рекомендации:")
        
        missing_secrets = [secret for secret, valid in status_after.items() if not valid]
        if missing_secrets:
            print(f"  📝 Необходимо настроить секреты: {', '.join(missing_secrets)}")
            print(f"  📁 Отредактируйте файл .env в корне проекта")
            
            if "GOOGLE_API_KEY" in missing_secrets:
                print(f"  🔑 Google API Key: https://makersuite.google.com/app/apikey")
            if "HF_TOKEN" in missing_secrets:
                print(f"  🤗 Hugging Face Token: https://huggingface.co/settings/tokens")
        else:
            print(f"  ✅ Все секреты настроены корректно!")
        
        # Проверка безопасности .gitignore
        gitignore_path = project_root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                gitignore_content = f.read()
            
            if ".env" in gitignore_content and "data/secrets/" in gitignore_content:
                print(f"  🔒 .gitignore настроен безопасно")
            else:
                print(f"  ⚠️  Проверьте .gitignore на предмет исключения секретов")
        
        print(f"\n🚀 Миграция завершена!")
        print(f"📖 Подробности см. в документации: config/secrets.py")
        
        return 0
        
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        logger.error("Убедитесь что установлены все зависимости: pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        logger.error(f"Ошибка миграции: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 