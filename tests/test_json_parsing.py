#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тест парсинга JSON ответов от LLM
"""

import sys
import os
sys.path.append('.')

from app.plugins.models.universal_llm_plugin import UniversalLLMPlugin

def test_json_parsing():
    """Тестирует парсинг различных форматов ответов JSON."""
    
    # Создаем тестовый плагин
    plugin = UniversalLLMPlugin('google', 'models/gemini-2.0-flash-exp')
    
    # Тестируем различные форматы ответов
    test_responses = [
        # Обычный JSON
        '{"sender": "ООО Тест", "total": 1000}',
        
        # JSON в markdown блоке
        '```json\n{"sender": "ООО Тест", "total": 1000}\n```',
        
        # JSON с текстом вокруг
        'Вот результат анализа:\n{"sender": "ООО Тест", "total": 1000}\nДанные извлечены успешно.',
        
        # Ошибка API
        'Request timed out.',
        
        # Сложный JSON
        '{"sender": "ООО \\"Рога и копыта\\"", "invoice_number": "001", "total": 15000.50, "currency": "RUB"}',
        
        # JSON с переносами строк
        '''```json
{
  "sender": "ООО Тест",
  "invoice_number": "001",
  "total": 1000,
  "currency": "RUB"
}
```''',
    ]
    
    print('Тестирование парсинга различных форматов ответов:')
    print('=' * 60)
    
    for i, response in enumerate(test_responses, 1):
        preview = response[:50].replace('\n', '\\n')
        print(f'\nТест {i}: {preview}...')
        try:
            result = plugin.parse_llm_response(response)
            print(f'✅ Результат: {result}')
        except Exception as e:
            print(f'❌ Ошибка: {e}')
        print('-' * 40)

if __name__ == "__main__":
    test_json_parsing() 