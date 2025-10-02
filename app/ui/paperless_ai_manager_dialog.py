"""
Диалог управления Paperless-AI расширенными функциями
Кастомные правила, статистика, обучение
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QTextEdit, QMessageBox,
    QTabWidget, QWidget, QComboBox, QListWidget, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime


class CustomRuleWidget(QWidget):
    """Виджет для создания/редактирования кастомного правила"""
    
    rule_saved = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Форма правила
        form_group = QGroupBox(self.tr("Параметры правила"))
        form_layout = QFormLayout()
        
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(self.tr("Название правила"))
        form_layout.addRow(self.tr("Название:"), self.name_edit)
        
        self.pattern_edit = QLineEdit()
        self.pattern_edit.setPlaceholderText(self.tr("Regex или ключевые слова"))
        form_layout.addRow(self.tr("Паттерн:"), self.pattern_edit)
        
        self.tags_edit = QLineEdit()
        self.tags_edit.setPlaceholderText(self.tr("тег1, тег2, тег3"))
        form_layout.addRow(self.tr("Теги (через запятую):"), self.tags_edit)
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(1.0)
        form_layout.addRow(self.tr("Уверенность:"), self.confidence_spin)
        
        self.enabled_check = QCheckBox(self.tr("Правило активно"))
        self.enabled_check.setChecked(True)
        form_layout.addRow("", self.enabled_check)
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        self.save_btn = QPushButton(self.tr("Сохранить правило"))
        self.save_btn.clicked.connect(self._save_rule)
        buttons_layout.addWidget(self.save_btn)
        
        self.clear_btn = QPushButton(self.tr("Очистить"))
        self.clear_btn.clicked.connect(self._clear_form)
        buttons_layout.addWidget(self.clear_btn)
        
        layout.addLayout(buttons_layout)
        
        # Примеры
        examples_group = QGroupBox(self.tr("Примеры паттернов"))
        examples_layout = QVBoxLayout()
        
        examples_text = QTextEdit()
        examples_text.setReadOnly(True)
        examples_text.setMaximumHeight(100)
        examples_text.setPlainText(
            "Точное совпадение: 'счет-фактура'\n"
            "Regex: '\\d{2}\\.\\d{2}\\.\\d{4}' (дата)\n"
            "Любое слово: '.*услуг.*' (содержит 'услуг')\n"
            "Начало строки: '^ООО' (начинается с 'ООО')"
        )
        examples_layout.addWidget(examples_text)
        
        examples_group.setLayout(examples_layout)
        layout.addWidget(examples_group)
    
    def _save_rule(self):
        """Сохраняет правило"""
        if not self.name_edit.text() or not self.pattern_edit.text():
            QMessageBox.warning(self, self.tr("Ошибка"), 
                               self.tr("Заполните название и паттерн"))
            return
        
        tags = [tag.strip() for tag in self.tags_edit.text().split(',') if tag.strip()]
        if not tags:
            QMessageBox.warning(self, self.tr("Ошибка"), 
                               self.tr("Укажите хотя бы один тег"))
            return
        
        rule_data = {
            "rule_id": f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": self.name_edit.text(),
            "pattern": self.pattern_edit.text(),
            "tags": tags,
            "confidence": self.confidence_spin.value(),
            "enabled": self.enabled_check.isChecked()
        }
        
        self.rule_saved.emit(rule_data)
        self._clear_form()
        
        QMessageBox.information(self, self.tr("Успех"), 
                               self.tr(f"Правило '{rule_data['name']}' сохранено!"))
    
    def _clear_form(self):
        """Очищает форму"""
        self.name_edit.clear()
        self.pattern_edit.clear()
        self.tags_edit.clear()
        self.confidence_spin.setValue(1.0)
        self.enabled_check.setChecked(True)


class StatisticsWidget(QWidget):
    """Виджет отображения статистики AI тегирования"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Общая статистика
        summary_group = QGroupBox(self.tr("Общая статистика"))
        summary_layout = QFormLayout()
        
        self.total_docs_label = QLabel("0")
        summary_layout.addRow(self.tr("Всего документов:"), self.total_docs_label)
        
        self.total_suggested_label = QLabel("0")
        summary_layout.addRow(self.tr("Предложено тегов:"), self.total_suggested_label)
        
        self.total_applied_label = QLabel("0")
        summary_layout.addRow(self.tr("Применено тегов:"), self.total_applied_label)
        
        self.acceptance_rate_label = QLabel("0%")
        summary_layout.addRow(self.tr("Процент принятия:"), self.acceptance_rate_label)
        
        self.session_duration_label = QLabel("0:00:00")
        summary_layout.addRow(self.tr("Время сессии:"), self.session_duration_label)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Топ тегов
        top_tags_group = QGroupBox(self.tr("Топ-10 тегов по эффективности"))
        top_tags_layout = QVBoxLayout()
        
        self.top_tags_table = QTableWidget()
        self.top_tags_table.setColumnCount(4)
        self.top_tags_table.setHorizontalHeaderLabels([
            self.tr("Тег"), 
            self.tr("Точность"), 
            self.tr("Применено"), 
            self.tr("Отклонено")
        ])
        self.top_tags_table.horizontalHeader().setStretchLastSection(True)
        self.top_tags_table.setMaximumHeight(250)
        
        top_tags_layout.addWidget(self.top_tags_table)
        top_tags_group.setLayout(top_tags_layout)
        layout.addWidget(top_tags_group)
        
        # Кнопки действий
        buttons_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton(self.tr("🔄 Обновить"))
        self.refresh_btn.clicked.connect(self.refresh_statistics)
        buttons_layout.addWidget(self.refresh_btn)
        
        self.export_btn = QPushButton(self.tr("📊 Экспорт статистики"))
        self.export_btn.clicked.connect(self.export_statistics)
        buttons_layout.addWidget(self.export_btn)
        
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
    
    def update_statistics(self, stats: Dict[str, Any]):
        """Обновляет отображаемую статистику"""
        try:
            self.total_docs_label.setText(str(stats.get("total_documents", 0)))
            self.total_suggested_label.setText(str(stats.get("total_tags_suggested", 0)))
            self.total_applied_label.setText(str(stats.get("total_tags_applied", 0)))
            
            acceptance_rate = stats.get("acceptance_rate", 0) * 100
            self.acceptance_rate_label.setText(f"{acceptance_rate:.1f}%")
            
            self.session_duration_label.setText(stats.get("session_duration", "0:00:00"))
            
            # Обновляем таблицу топ тегов
            tag_accuracy = stats.get("tag_accuracy", {})
            self.top_tags_table.setRowCount(0)
            
            # Сортируем по точности
            sorted_tags = sorted(
                tag_accuracy.items(), 
                key=lambda x: x[1].get("accuracy", 0), 
                reverse=True
            )[:10]
            
            for row, (tag_name, tag_stats) in enumerate(sorted_tags):
                self.top_tags_table.insertRow(row)
                
                # Тег
                self.top_tags_table.setItem(row, 0, QTableWidgetItem(tag_name))
                
                # Точность
                accuracy = tag_stats.get("accuracy", 0) * 100
                accuracy_item = QTableWidgetItem(f"{accuracy:.1f}%")
                self.top_tags_table.setItem(row, 1, accuracy_item)
                
                # Применено
                applied = QTableWidgetItem(str(tag_stats.get("applied", 0)))
                self.top_tags_table.setItem(row, 2, applied)
                
                # Отклонено
                rejected = QTableWidgetItem(str(tag_stats.get("rejected", 0)))
                self.top_tags_table.setItem(row, 3, rejected)
            
            self.top_tags_table.resizeColumnsToContents()
            
        except Exception as e:
            logging.error(f"Ошибка обновления статистики: {e}")
    
    def refresh_statistics(self):
        """Обновляет статистику (переопределяется родителем)"""
        pass
    
    def export_statistics(self):
        """Экспортирует статистику (переопределяется родителем)"""
        pass


class PaperlessAIManagerDialog(QDialog):
    """Диалог управления расширенными функциями Paperless-AI"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Управление Paperless-AI"))
        self.setMinimumSize(800, 700)
        
        self.ai_plugin = None
        self.custom_rules = []
        
        self._init_ui()
        self._load_settings()
    
    def _init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout(self)
        
        # Вкладки
        tabs = QTabWidget()
        
        # Вкладка кастомных правил
        tabs.addTab(self._create_rules_tab(), self.tr("📋 Кастомные правила"))
        
        # Вкладка статистики
        tabs.addTab(self._create_statistics_tab(), self.tr("📊 Статистика"))
        
        # Вкладка обучения
        tabs.addTab(self._create_learning_tab(), self.tr("🎓 Обучение"))
        
        layout.addWidget(tabs)
        
        # Кнопки
        buttons_layout = QHBoxLayout()
        
        buttons_layout.addStretch()
        
        self.close_btn = QPushButton(self.tr("Закрыть"))
        self.close_btn.clicked.connect(self.accept)
        buttons_layout.addWidget(self.close_btn)
        
        layout.addLayout(buttons_layout)
    
    def _create_rules_tab(self) -> QWidget:
        """Создает вкладку кастомных правил"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Создание правила
        self.rule_widget = CustomRuleWidget()
        self.rule_widget.rule_saved.connect(self._on_rule_saved)
        layout.addWidget(self.rule_widget)
        
        # Список существующих правил
        rules_group = QGroupBox(self.tr("Существующие правила"))
        rules_layout = QVBoxLayout()
        
        self.rules_table = QTableWidget()
        self.rules_table.setColumnCount(5)
        self.rules_table.setHorizontalHeaderLabels([
            self.tr("Название"), 
            self.tr("Паттерн"), 
            self.tr("Теги"), 
            self.tr("Активно"),
            self.tr("Действия")
        ])
        self.rules_table.horizontalHeader().setStretchLastSection(True)
        
        rules_layout.addWidget(self.rules_table)
        rules_group.setLayout(rules_layout)
        layout.addWidget(rules_group)
        
        return widget
    
    def _create_statistics_tab(self) -> QWidget:
        """Создает вкладку статистики"""
        self.statistics_widget = StatisticsWidget()
        self.statistics_widget.refresh_statistics = self._refresh_statistics
        self.statistics_widget.export_statistics = self._export_statistics
        return self.statistics_widget
    
    def _create_learning_tab(self) -> QWidget:
        """Создает вкладку обучения"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Информация об обучении
        info_group = QGroupBox(self.tr("Информация об обучении"))
        info_layout = QVBoxLayout()
        
        info_text = QLabel(
            self.tr(
                "Система собирает данные о предложенных и примененных тегах для улучшения "
                "качества AI тегирования. Эти данные можно экспортировать для обучения "
                "кастомной модели или анализа."
            )
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Настройки обучения
        settings_group = QGroupBox(self.tr("Настройки обучения"))
        settings_layout = QFormLayout()
        
        self.enable_learning_check = QCheckBox(self.tr("Включить сбор данных для обучения"))
        self.enable_learning_check.setChecked(True)
        settings_layout.addRow("", self.enable_learning_check)
        
        self.learning_history_limit = QSpinBox()
        self.learning_history_limit.setRange(100, 10000)
        self.learning_history_limit.setValue(1000)
        settings_layout.addRow(self.tr("Размер истории:"), self.learning_history_limit)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Действия
        actions_group = QGroupBox(self.tr("Действия"))
        actions_layout = QVBoxLayout()
        
        export_learning_btn = QPushButton(self.tr("📤 Экспорт данных для обучения"))
        export_learning_btn.clicked.connect(self._export_learning_data)
        actions_layout.addWidget(export_learning_btn)
        
        clear_history_btn = QPushButton(self.tr("🗑️ Очистить историю обучения"))
        clear_history_btn.clicked.connect(self._clear_learning_history)
        actions_layout.addWidget(clear_history_btn)
        
        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)
        
        # Статистика обучения
        learning_stats_group = QGroupBox(self.tr("Статистика обучающих данных"))
        learning_stats_layout = QFormLayout()
        
        self.history_size_label = QLabel("0")
        learning_stats_layout.addRow(self.tr("Записей в истории:"), self.history_size_label)
        
        self.unique_tags_label = QLabel("0")
        learning_stats_layout.addRow(self.tr("Уникальных тегов:"), self.unique_tags_label)
        
        learning_stats_group.setLayout(learning_stats_layout)
        layout.addWidget(learning_stats_group)
        
        layout.addStretch()
        
        return widget
    
    def _on_rule_saved(self, rule_data: Dict[str, Any]):
        """Обработчик сохранения правила"""
        try:
            if self.ai_plugin and hasattr(self.ai_plugin, 'add_custom_rule'):
                from app.plugins.integrations.paperless_ai_advanced import CustomTaggingRule
                
                rule = CustomTaggingRule(
                    rule_id=rule_data["rule_id"],
                    name=rule_data["name"],
                    pattern=rule_data["pattern"],
                    tags=rule_data["tags"],
                    enabled=rule_data["enabled"],
                    confidence=rule_data["confidence"]
                )
                
                if self.ai_plugin.add_custom_rule(rule):
                    self._refresh_rules_table()
            else:
                # Сохраняем локально
                self.custom_rules.append(rule_data)
                self._refresh_rules_table()
                
                # Уведомление о недоступности расширенных функций
                if not self.ai_plugin:
                    QMessageBox.information(
                        self, 
                        self.tr("Информация"),
                        self.tr("Правило сохранено локально. Для полной функциональности подключите Paperless-AI Advanced плагин.")
                    )
                
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка сохранения правила: {e}"))
            logging.error(f"Ошибка сохранения правила: {e}", exc_info=True)
    
    def _refresh_rules_table(self):
        """Обновляет таблицу правил"""
        try:
            self.rules_table.setRowCount(0)
            
            rules_to_display = []
            if self.ai_plugin:
                rules_to_display = [rule.to_dict() for rule in self.ai_plugin.custom_rules.values()]
            else:
                rules_to_display = self.custom_rules
            
            for row, rule in enumerate(rules_to_display):
                self.rules_table.insertRow(row)
                
                # Название
                self.rules_table.setItem(row, 0, QTableWidgetItem(rule["name"]))
                
                # Паттерн
                self.rules_table.setItem(row, 1, QTableWidgetItem(rule["pattern"]))
                
                # Теги
                tags_str = ", ".join(rule["tags"])
                self.rules_table.setItem(row, 2, QTableWidgetItem(tags_str))
                
                # Активно
                enabled_str = "✓" if rule["enabled"] else "✗"
                self.rules_table.setItem(row, 3, QTableWidgetItem(enabled_str))
                
                # Действия
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(2, 2, 2, 2)
                
                delete_btn = QPushButton("🗑")
                delete_btn.setMaximumWidth(40)
                delete_btn.clicked.connect(lambda checked, r=rule["rule_id"]: self._delete_rule(r))
                actions_layout.addWidget(delete_btn)
                
                self.rules_table.setCellWidget(row, 4, actions_widget)
            
            self.rules_table.resizeColumnsToContents()
            
        except Exception as e:
            logging.error(f"Ошибка обновления таблицы правил: {e}")
    
    def _delete_rule(self, rule_id: str):
        """Удаляет правило"""
        reply = QMessageBox.question(
            self, 
            self.tr("Подтверждение"), 
            self.tr("Удалить это правило?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if self.ai_plugin:
                    self.ai_plugin.remove_custom_rule(rule_id)
                else:
                    self.custom_rules = [r for r in self.custom_rules if r["rule_id"] != rule_id]
                
                self._refresh_rules_table()
                
            except Exception as e:
                QMessageBox.critical(self, self.tr("Ошибка"), 
                                    self.tr(f"Ошибка удаления правила: {e}"))
    
    def _refresh_statistics(self):
        """Обновляет статистику"""
        try:
            if self.ai_plugin:
                stats = self.ai_plugin.get_statistics()
                self.statistics_widget.update_statistics(stats)
            
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка обновления статистики: {e}"))
    
    def _export_statistics(self):
        """Экспортирует статистику"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("Экспорт статистики"),
                f"paperless_ai_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                self.tr("JSON файлы (*.json)")
            )
            
            if file_path:
                if self.ai_plugin:
                    stats = self.ai_plugin.get_statistics()
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(stats, f, indent=2, ensure_ascii=False)
                    
                    QMessageBox.information(self, self.tr("Успех"), 
                                           self.tr(f"Статистика экспортирована в {file_path}"))
                
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка экспорта статистики: {e}"))
    
    def _export_learning_data(self):
        """Экспортирует данные для обучения"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                self.tr("Экспорт данных для обучения"),
                f"learning_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                self.tr("JSON файлы (*.json)")
            )
            
            if file_path and self.ai_plugin:
                if self.ai_plugin.export_learning_data(Path(file_path)):
                    QMessageBox.information(self, self.tr("Успех"), 
                                           self.tr(f"Данные экспортированы в {file_path}"))
                
        except Exception as e:
            QMessageBox.critical(self, self.tr("Ошибка"), 
                                self.tr(f"Ошибка экспорта данных: {e}"))
    
    def _clear_learning_history(self):
        """Очищает историю обучения"""
        reply = QMessageBox.question(
            self,
            self.tr("Подтверждение"),
            self.tr("Очистить всю историю обучения? Это действие нельзя отменить."),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if self.ai_plugin:
                    self.ai_plugin.suggestion_history.clear()
                    QMessageBox.information(self, self.tr("Успех"), 
                                           self.tr("История обучения очищена"))
                
            except Exception as e:
                QMessageBox.critical(self, self.tr("Ошибка"), 
                                    self.tr(f"Ошибка очистки истории: {e}"))
    
    def _load_settings(self):
        """Загружает настройки"""
        try:
            self._refresh_rules_table()
            self._refresh_statistics()
            
        except Exception as e:
            logging.error(f"Ошибка загрузки настроек: {e}")
    
    def set_ai_plugin(self, plugin):
        """Устанавливает плагин AI для использования"""
        self.ai_plugin = plugin
        self._refresh_rules_table()
        self._refresh_statistics()

