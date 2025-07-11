# 🔒 Безопасность InvoiceGemini

## ⚠️ ВАЖНОЕ УВЕДОМЛЕНИЕ О БЕЗОПАСНОСТИ

**Дата:** Декабрь 2024  
**Статус:** Устранено  
**Приоритет:** КРИТИЧЕСКИЙ  

### Проблема
В предыдущих версиях репозитория в git истории могли содержаться следующие чувствительные файлы:
- `data/security/.encryption.key` - Главный ключ шифрования
- `data/security/.secrets.enc` - Зашифрованные секреты
- `data/security/secrets_backup_*.enc` - Файлы резервных копий секретов
- `data/cache/cache_index.json` - Кэш который мог содержать чувствительные данные

### Принятые меры
1. ✅ Удалены все чувствительные файлы из git индекса
2. ✅ Обновлен .gitignore с расширенными правилами безопасности
3. ✅ Создан этот документ по безопасности

### Рекомендации по безопасности

#### Для разработчиков
1. **Немедленно ротируйте все API ключи:**
   - Google Gemini API ключи
   - OpenAI API ключи
   - Anthropic API ключи
   - Любые другие облачные сервисы

2. **Пересоздайте ключи шифрования:**
   ```bash
   # Удалите старые ключи
   rm -rf data/security/.encryption.key
   
   # Запустите приложение - оно создаст новые ключи автоматически
   python main.py
   ```

3. **Проверьте историю git:**
   ```bash
   # Найти файлы секретов в истории
   git log --name-only --grep="secret\|key\|password"
   ```

#### Для пользователей
1. **Обновите до последней версии** из main ветки
2. **Повторно введите все API ключи** в настройках приложения
3. **Удалите старые данные кэша:**
   ```bash
   rm -rf data/cache/*
   rm -rf data/temp/*
   ```

## Настройка безопасности

### Защищенные директории
Следующие директории автоматически исключены из git:

```
data/security/          # Ключи шифрования и секреты
data/secrets/           # Дополнительные секреты
data/cache/             # Кэш данные
logs/                   # Логи приложения
*.key, *.enc, *.pem     # Файлы ключей и сертификатов
*secret*, *credential*  # Любые файлы с секретами
.env*                   # Файлы окружения
```

### Безопасное хранение секретов
InvoiceGemini использует многоуровневую защиту:

1. **Шифрование AES-256** для всех API ключей
2. **Отдельный ключ шифрования** хранящийся локально
3. **Автоматическое создание резервных копий** зашифрованных секретов
4. **Проверка целостности** при загрузке секретов

### Конфигурация .gitignore
```gitignore
# SECURITY - CRITICAL: SECRET DATA EXCLUSION
data/security/
*.key
*.enc
*.pem
*secret*
*credential*
*password*
*token*
*api_key*
.secrets*
.env*
data/cache/
*.backup
```

## Действия при компрометации

### Если секреты попали в git:
1. **НЕ** пытайтесь удалить коммиты самостоятельно
2. **Немедленно** ротируйте все API ключи
3. **Измените** мастер-пароль шифрования
4. **Уведомите** команду разработки

### Экстренные действия:
```bash
# Остановить приложение
pkill -f "python.*main.py"

# Очистить временные данные
rm -rf data/temp/*
rm -rf data/cache/*

# Пересоздать ключи шифрования
rm data/security/.encryption.key
python -c "from app.security.crypto_manager import CryptoManager; CryptoManager().initialize()"
```

## Отчеты о уязвимостях

Если вы обнаружили уязвимость безопасности:

1. **НЕ** создавайте публичные issues
2. **Свяжитесь** с командой разработки приватно
3. **Предоставьте** детальное описание проблемы
4. **Дождитесь** подтверждения получения

## История изменений безопасности

### Декабрь 2024
- ✅ Удалены секретные файлы из git репозитория
- ✅ Расширены правила .gitignore для безопасности
- ✅ Создан документ SECURITY.md
- ✅ Добавлены рекомендации по ротации ключей

### Ноябрь 2024
- ✅ Внедрена система шифрования AES-256
- ✅ Добавлено автоматическое резервное копирование секретов
- ✅ Реализована проверка целостности секретов

---

**🔐 Безопасность - это общая ответственность. Следуйте этим рекомендациям для защиты вашей инфраструктуры.** 