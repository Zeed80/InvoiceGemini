#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Тесты для UniversalLLMPlugin в части работы с Ollama."""

import json
import os
import pytest

from app.plugins.models.universal_llm_plugin import UniversalLLMPlugin


@pytest.fixture
def plugin_mock(monkeypatch):
    """Создает экземпляр плагина с подменой сетевых вызовов и мока OCR."""

    plugin = UniversalLLMPlugin(provider_name="ollama", model_name="gemma3:4b")
    plugin.is_loaded = True  # Обходим фактическую загрузку клиента

    monkeypatch.setattr(plugin, "extract_text_from_image", lambda *_, **__: "")
    return plugin


def test_prompt_is_neutral(monkeypatch):
    """Убеждаемся, что промпт не содержит зашитых значений и формируется адаптивно."""

    plugin = UniversalLLMPlugin(provider_name="ollama", model_name="gemma3:4b")
    plugin.is_loaded = True

    def fake_generate_response(prompt, *_, **__):
        return prompt

    monkeypatch.setattr(plugin, "generate_response", fake_generate_response)

    prompt = plugin.create_invoice_prompt(use_adaptive=False)
    assert "АО \"ПТС\"" not in prompt
    assert "receiver_name" not in prompt
    assert "Use" in prompt or "use" in prompt


def test_parse_ollama_response(plugin_mock, tmp_path):
    """Проверяем, что базовый парсер корректно извлекает данные из JSON."""

    response_data = {
        "sender": "ООО Тест",
        "invoice_number": "A-102",
        "invoice_date": "01.09.2025",
        "total": "1000.00",
        "amount_no_vat": "800.00",
        "vat_percent": "20",
        "currency": "RUB",
        "category": "Услуги",
        "description": "Консультация",
        "note": ""
    }

    def fake_generate_response(*_, **__):
        return json.dumps(response_data)

    plugin_mock.generate_response = fake_generate_response

    fake_image = tmp_path / "invoice.png"
    fake_image.write_bytes(b"fake")

    result = plugin_mock.process_image(str(fake_image), custom_prompt=None)
    assert result["sender"] == "ООО Тест"
    assert result["invoice_number"] == "A-102"
    assert result["currency"] == "RUB"


def test_parse_ollama_bad_response(plugin_mock, tmp_path):
    """Fallback: пустой или фиктивный ответ приводит к сообщению об ошибке."""

    plugin_mock.generate_response = lambda *_, **__: '{"sender": "12345", "invoice_number": "12345"}'

    fake_image = tmp_path / "invoice.png"
    fake_image.write_bytes(b"fake")

    result = plugin_mock.process_image(str(fake_image), custom_prompt=None)
    assert "error" in result
    assert "raw_response" in result
