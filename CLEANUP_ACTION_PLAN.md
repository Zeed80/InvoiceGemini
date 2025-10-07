# 🧹 План очистки проекта InvoiceGemini

## 🎯 Цель
Привести проект в порядок, удалив мусор и организовав структуру файлов.

---

## 📋 Что делать СЕЙЧАС

### Шаг 1: Создать структуру docs/ (5 мин)

```bash
mkdir -p docs/{architecture,development,features,reports/implementation,user-guides}
```

### Шаг 2: Переместить документацию (10 мин)

```bash
# Architecture
mv SYSTEM_ARCHITECTURE_ANALYSIS.md docs/architecture/
mv REFACTORING_GUIDE.md docs/architecture/
mv CODE_AUDIT_REPORT.md docs/architecture/
mv AUDIT_REPORT.md docs/architecture/
mv AUDIT_COMPLETE.md docs/architecture/

# Development
mv DEVELOPMENT_PLAN.md docs/development/
mv DEVELOPMENT_SUMMARY.md docs/development/
mv BUGFIX_PLAN.md docs/development/
mv ERROR_FIXING_PLAN.md docs/development/
mv LORA_PLAN.md docs/development/
mv LORA_ENHANCEMENT_PLAN.md docs/development/
mv HIGH_MEDIUM_PRIORITY_IMPLEMENTATION.md docs/development/

# Features
mv PAPERLESS_INTEGRATION_GUIDE.md docs/features/
mv PAPERLESS_AI_ADVANCED_GUIDE.md docs/features/
mv PLUGIN_SYSTEM_GUIDE.md docs/features/
mv TROCR_AUTOMATION_GUIDE.md docs/features/
mv TROCR_DATASET_GUIDE.md docs/features/
mv TROCR_INTEGRATION.md docs/features/
mv TROCR_FINAL_AUDIT_REPORT.md docs/features/
mv DONUT_HIGH_ACCURACY_TRAINING_GUIDE.md docs/features/
mv OPTIMIZATION_GUIDE.md docs/features/

# Implementation Reports
mv IMPLEMENTATION_SUMMARY.md docs/reports/implementation/
mv IMPLEMENTATION_REPORT.md docs/reports/implementation/
mv PAPERLESS_INTEGRATION_SUMMARY.md docs/reports/implementation/
mv PAPERLESS_AI_INTEGRATION_SUMMARY.md docs/reports/implementation/
mv PAPERLESS_FULL_INTEGRATION_COMPLETE.md docs/reports/implementation/
mv PHASE3_IMPLEMENTATION_SUMMARY.md docs/reports/implementation/
mv PLUGIN_SYSTEM_UPDATE_SUMMARY.md docs/reports/implementation/
mv OPTIMIZATION_SUMMARY.md docs/reports/implementation/
mv TROCR_AUTOMATION_SUMMARY.md docs/reports/implementation/
mv PDF_ANALYZER_INTEGRATION_SUMMARY.md docs/reports/implementation/
mv MAIN_PY_IMPROVEMENTS_SUMMARY.md docs/reports/implementation/
mv INTEGRATION_SUMMARY.md docs/reports/implementation/
mv BUGFIX_SUMMARY.md docs/reports/implementation/
mv GEMINI_FIX_SUMMARY.md docs/reports/implementation/
mv CLOUD_LLM_FIX_SUMMARY.md docs/reports/implementation/

# User Guides
mv PAPERLESS_USER_GUIDE.md docs/user-guides/
mv QUICK_START_INTEGRATIONS.md docs/user-guides/
mv INTEGRATION_EXAMPLES.md docs/user-guides/
```

### Шаг 3: Удалить backup файлы (2 мин)

```bash
rm main_original_backup.py
rm main_refactored.py
rm app/main_window_backup.py
rm app/training/*.backup_*
rm InvoiceGemini.7z
```

### Шаг 4: Организовать demo/test (5 мин)

```bash
# Создать папки
mkdir -p examples tests

# Переместить demo
mv demo_*.py examples/
mv init_prompts.py examples/
mv generate_translations.py examples/

# Переместить тесты
mv test_*.py tests/
mv debug_runner.py tests/

# Удалить логи
rm *.log
```

### Шаг 5: Обновить .gitignore (1 мин)

```bash
# Добавить в .gitignore
echo "" >> .gitignore
echo "# Logs" >> .gitignore
echo "*.log" >> .gitignore
echo "logs/" >> .gitignore
echo "" >> .gitignore
echo "# Archives" >> .gitignore
echo "*.7z" >> .gitignore
echo "*.zip" >> .gitignore
echo "*.tar.gz" >> .gitignore
echo "" >> .gitignore
echo "# Backups" >> .gitignore
echo "*.backup" >> .gitignore
echo "*_backup.py" >> .gitignore
echo "*_original_backup.py" >> .gitignore
```

### Шаг 6: Создать README для docs/ (3 мин)

```bash
cat > docs/README.md << 'EOF'
# Документация InvoiceGemini

## Структура

- **architecture/** - Архитектурные решения и анализ
- **development/** - Планы разработки
- **features/** - Руководства по функциям
- **reports/implementation/** - Отчёты о реализации
- **user-guides/** - Руководства пользователя

## Основные документы

### Для разработчиков
- [Архитектура](architecture/SYSTEM_ARCHITECTURE_ANALYSIS.md)
- [Руководство по рефакторингу](architecture/REFACTORING_GUIDE.md)
- [План разработки](development/DEVELOPMENT_PLAN.md)

### Для пользователей
- [Быстрый старт](user-guides/QUICK_START_INTEGRATIONS.md)
- [Paperless интеграция](user-guides/PAPERLESS_USER_GUIDE.md)
- [Примеры](user-guides/INTEGRATION_EXAMPLES.md)

### Функции
- [Система плагинов](features/PLUGIN_SYSTEM_GUIDE.md)
- [TrOCR автоматизация](features/TROCR_AUTOMATION_GUIDE.md)
- [Paperless AI](features/PAPERLESS_AI_ADVANCED_GUIDE.md)
EOF
```

### Шаг 7: Обновить корневой README (2 мин)

Добавить в конец README.md:

```markdown
## 📚 Документация

Вся документация перенесена в папку [docs/](docs/).

### Быстрые ссылки:
- 🚀 [Быстрый старт](docs/user-guides/QUICK_START_INTEGRATIONS.md)
- 📖 [Руководство пользователя Paperless](docs/user-guides/PAPERLESS_USER_GUIDE.md)
- 🔌 [Система плагинов](docs/features/PLUGIN_SYSTEM_GUIDE.md)
- 🏗️ [Архитектура](docs/architecture/SYSTEM_ARCHITECTURE_ANALYSIS.md)

Полный список документов: [docs/README.md](docs/README.md)
```

---

## ✅ Результат

### До:
```
InvoiceGemini/
├── 40+ MD файлов в корне 😱
├── backup файлы
├── demo файлы
├── test файлы
├── логи
└── архивы
```

### После:
```
InvoiceGemini/
├── docs/                    # Вся документация
│   ├── architecture/
│   ├── development/
│   ├── features/
│   ├── reports/
│   └── user-guides/
├── examples/               # Demo файлы
├── tests/                  # Тесты
├── app/                    # Код приложения
├── README.md              # Основной README
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
└── requirements.txt
```

---

## 🎯 Время выполнения

**Всего:** ~30 минут

| Шаг | Время | Описание |
|-----|-------|----------|
| 1 | 5 мин | Создать структуру docs/ |
| 2 | 10 мин | Переместить MD файлы |
| 3 | 2 мин | Удалить backup |
| 4 | 5 мин | Организовать demo/test |
| 5 | 1 мин | Обновить .gitignore |
| 6 | 3 мин | Создать docs/README.md |
| 7 | 2 мин | Обновить корневой README |
| 8 | 2 мин | Git commit |

---

## 📝 Команды для копипаста

### Полный скрипт очистки:

```bash
#!/bin/bash
# cleanup.sh - Скрипт очистки проекта InvoiceGemini

echo "🧹 Начинаем очистку проекта..."

# 1. Создать структуру docs/
echo "📁 Создаём структуру docs/..."
mkdir -p docs/{architecture,development,features,reports/implementation,user-guides}

# 2. Переместить документацию
echo "📚 Перемещаем документацию..."

# Architecture
mv SYSTEM_ARCHITECTURE_ANALYSIS.md docs/architecture/ 2>/dev/null
mv REFACTORING_GUIDE.md docs/architecture/ 2>/dev/null
mv CODE_AUDIT_REPORT.md docs/architecture/ 2>/dev/null
mv AUDIT_REPORT.md docs/architecture/ 2>/dev/null
mv AUDIT_COMPLETE.md docs/architecture/ 2>/dev/null

# Development
mv DEVELOPMENT_PLAN.md docs/development/ 2>/dev/null
mv DEVELOPMENT_SUMMARY.md docs/development/ 2>/dev/null
mv BUGFIX_PLAN.md docs/development/ 2>/dev/null
mv ERROR_FIXING_PLAN.md docs/development/ 2>/dev/null
mv LORA_PLAN.md docs/development/ 2>/dev/null
mv LORA_ENHANCEMENT_PLAN.md docs/development/ 2>/dev/null
mv HIGH_MEDIUM_PRIORITY_IMPLEMENTATION.md docs/development/ 2>/dev/null

# Features
mv PAPERLESS_INTEGRATION_GUIDE.md docs/features/ 2>/dev/null
mv PAPERLESS_AI_ADVANCED_GUIDE.md docs/features/ 2>/dev/null
mv PLUGIN_SYSTEM_GUIDE.md docs/features/ 2>/dev/null
mv TROCR_AUTOMATION_GUIDE.md docs/features/ 2>/dev/null
mv TROCR_DATASET_GUIDE.md docs/features/ 2>/dev/null
mv TROCR_INTEGRATION.md docs/features/ 2>/dev/null
mv TROCR_FINAL_AUDIT_REPORT.md docs/features/ 2>/dev/null
mv DONUT_HIGH_ACCURACY_TRAINING_GUIDE.md docs/features/ 2>/dev/null
mv OPTIMIZATION_GUIDE.md docs/features/ 2>/dev/null

# Implementation Reports
mv IMPLEMENTATION_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv IMPLEMENTATION_REPORT.md docs/reports/implementation/ 2>/dev/null
mv PAPERLESS_INTEGRATION_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv PAPERLESS_AI_INTEGRATION_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv PAPERLESS_FULL_INTEGRATION_COMPLETE.md docs/reports/implementation/ 2>/dev/null
mv PHASE3_IMPLEMENTATION_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv PLUGIN_SYSTEM_UPDATE_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv OPTIMIZATION_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv TROCR_AUTOMATION_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv PDF_ANALYZER_INTEGRATION_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv MAIN_PY_IMPROVEMENTS_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv INTEGRATION_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv BUGFIX_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv GEMINI_FIX_SUMMARY.md docs/reports/implementation/ 2>/dev/null
mv CLOUD_LLM_FIX_SUMMARY.md docs/reports/implementation/ 2>/dev/null

# User Guides
mv PAPERLESS_USER_GUIDE.md docs/user-guides/ 2>/dev/null
mv QUICK_START_INTEGRATIONS.md docs/user-guides/ 2>/dev/null
mv INTEGRATION_EXAMPLES.md docs/user-guides/ 2>/dev/null

# 3. Удалить backup
echo "🗑️  Удаляем backup файлы..."
rm -f main_original_backup.py 2>/dev/null
rm -f main_refactored.py 2>/dev/null
rm -f app/main_window_backup.py 2>/dev/null
rm -f app/training/*.backup_* 2>/dev/null
rm -f InvoiceGemini.7z 2>/dev/null

# 4. Организовать demo/test
echo "📦 Организуем demo и тесты..."
mkdir -p examples tests

mv demo_*.py examples/ 2>/dev/null
mv init_prompts.py examples/ 2>/dev/null
mv generate_translations.py examples/ 2>/dev/null

mv test_*.py tests/ 2>/dev/null
mv debug_runner.py tests/ 2>/dev/null

# 5. Удалить логи
echo "🧼 Удаляем логи..."
rm -f *.log 2>/dev/null

echo "✅ Очистка завершена!"
echo "📊 Проверьте результат:"
echo "   - docs/ создана и заполнена"
echo "   - backup файлы удалены"
echo "   - demo → examples/"
echo "   - tests → tests/"
echo ""
echo "💡 Не забудьте:"
echo "   1. Создать docs/README.md"
echo "   2. Обновить корневой README.md"
echo "   3. Сделать git commit"
```

---

## 🚀 Следующие шаги (после очистки)

1. **Git commit:**
```bash
git add .
git commit -m "chore: reorganize project structure and cleanup files

- Move all documentation to docs/ folder
- Remove backup and demo files
- Organize tests and examples
- Update .gitignore
- Add docs/README.md"
```

2. **Проверить проект:**
```bash
# Запустить приложение
python main.py

# Проверить импорты
python -c "import app"
```

3. **Следующий этап:**
- Исправить критические баги (см. PROJECT_ANALYSIS_REPORT.md)
- Добавить тесты
- Настроить CI/CD

---

*Создано: 02.10.2024*  
*Время выполнения: ~30 минут*  
*Статус: Готово к выполнению*

