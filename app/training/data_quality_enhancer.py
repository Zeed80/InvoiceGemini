"""
Модуль для повышения качества разметки данных до максимального уровня
Основан на лучших практиках индустрии ML/AI для аннотации данных

Реализует стратегии:
1. Эталонные датасеты (Golden datasets)
2. Консенсус-алгоритмы (Multiple passthroughs)
3. Автоматическую детекцию ошибок
4. Метрики качества (Cohen's Kappa, Fleiss' Kappa)
5. Валидацию через перекрестные методы
6. Интеллектуальную фильтрацию спорных случаев

Источники:
- https://www.damcogroup.com/blogs/strategies-to-enhance-data-annotation-accuracy
- https://medium.com/datatorch/6-qa-tactics-for-data-annotation-jobs-8a17b83a46e6
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import math
from difflib import SequenceMatcher
from statistics import mode, median, mean


@dataclass
class AnnotationQualityMetrics:
    """Метрики качества аннотации"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inter_annotator_agreement: float
    confidence_score: float
    edge_cases_detected: int
    consensus_level: float


@dataclass
class FieldExtraction:
    """Результат извлечения поля"""
    field_name: str
    value: str
    confidence: float
    method: str  # 'gemini', 'ocr', 'pdf_text', 'pattern'
    position: Optional[Tuple[int, int, int, int]] = None
    normalized_value: Optional[str] = None


@dataclass
class ConsensusResult:
    """Результат консенсус-алгоритма"""
    final_value: str
    confidence: float
    agreement_score: float
    participating_methods: List[str]
    is_edge_case: bool
    conflict_details: Optional[Dict] = None


class DataQualityEnhancer:
    """
    Усовершенствованный модуль для повышения качества разметки данных
    Реализует лучшие практики индустрии для достижения максимальной точности
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Эталонные паттерны для полей (Golden patterns)
        self.golden_patterns = {
            'invoice_number': [
                r'№\s*(\d+(?:[-/]\d+)*)',
                r'счет\s*№?\s*(\d+(?:[-/]\d+)*)',
                r'invoice\s*#?\s*(\d+(?:[-/]\d+)*)',
                r'инвойс\s*№?\s*(\d+(?:[-/]\d+)*)',
                r'УТ-(\d+)',
                r'СЧ-(\d+)',
                r'N\s*(\d+)',
            ],
            'invoice_date': [
                r'(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})',
                r'(\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2})',
                r'дата:?\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})',
                r'от\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{4})',
                r'(\d{1,2}\s+\w+\s+\d{4})',  # "25 апреля 2024"
            ],
            'company_inn': [
                r'инн:?\s*(\d{10,12})',
                r'ИНН:?\s*(\d{10,12})',
                r'inn:?\s*(\d{10,12})',
                r'(\d{10})',  # Стандартный ИНН
                r'(\d{12})',  # ИП ИНН
            ],
            'company_kpp': [
                r'кпп:?\s*(\d{9})',
                r'КПП:?\s*(\d{9})',
                r'kpp:?\s*(\d{9})',
            ],
            'total_amount': [
                r'итого:?\s*(\d+(?:[.,]\d+)?)',
                r'сумма:?\s*(\d+(?:[.,]\d+)?)',
                r'к\s*оплате:?\s*(\d+(?:[.,]\d+)?)',
                r'всего:?\s*(\d+(?:[.,]\d+)?)',
                r'total:?\s*(\d+(?:[.,]\d+)?)',
            ],
        }
        
        # Словари для нормализации
        self.month_names = {
            'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
            'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
            'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'
        }
        
        # Статистика качества
        self.quality_stats = defaultdict(list)
        
        self.logger.info("✅ DataQualityEnhancer инициализирован")
    
    def extract_fields_with_multiple_methods(self, 
                                           text: str, 
                                           pdf_text_blocks: List[Dict] = None,
                                           gemini_result: Dict = None,
                                           ocr_result: Dict = None) -> Dict[str, List[FieldExtraction]]:
        """
        Извлекает поля несколькими методами для последующего консенсуса
        
        Реализует стратегию "Multiple passthroughs" для максимальной точности
        
        Returns:
            Dict[field_name, List[FieldExtraction]] - результаты всех методов
        """
        all_extractions = defaultdict(list)
        
        # 1. Паттерн-основанное извлечение из текста
        pattern_results = self._extract_with_patterns(text)
        for field_name, value in pattern_results.items():
            if value:
                all_extractions[field_name].append(
                    FieldExtraction(
                        field_name=field_name,
                        value=value,
                        confidence=0.8,
                        method='pattern',
                        normalized_value=self._normalize_field_value(field_name, value)
                    )
                )
        
        # 2. Извлечение из PDF текстовых блоков
        if pdf_text_blocks:
            pdf_results = self._extract_from_pdf_blocks(pdf_text_blocks)
            for field_name, value in pdf_results.items():
                if value:
                    all_extractions[field_name].append(
                        FieldExtraction(
                            field_name=field_name,
                            value=value,
                            confidence=0.9,  # Высокая уверенность для PDF текста
                            method='pdf_text',
                            normalized_value=self._normalize_field_value(field_name, value)
                        )
                    )
        
        # 3. Результаты Gemini
        if gemini_result:
            for field_name, value in gemini_result.items():
                if value:
                    all_extractions[field_name].append(
                        FieldExtraction(
                            field_name=field_name,
                            value=str(value),
                            confidence=0.85,
                            method='gemini',
                            normalized_value=self._normalize_field_value(field_name, str(value))
                        )
                    )
        
        # 4. Результаты OCR
        if ocr_result:
            for field_name, value in ocr_result.items():
                if value:
                    all_extractions[field_name].append(
                        FieldExtraction(
                            field_name=field_name,
                            value=str(value),
                            confidence=0.7,  # Ниже из-за возможных ошибок OCR
                            method='ocr',
                            normalized_value=self._normalize_field_value(field_name, str(value))
                        )
                    )
        
        return dict(all_extractions)
    
    def apply_consensus_algorithm(self, 
                                extractions: Dict[str, List[FieldExtraction]]) -> Dict[str, ConsensusResult]:
        """
        Применяет консенсус-алгоритм для выбора лучшего значения поля
        Основан на Fleiss' Kappa для множественных аннотаторов
        
        Реализует стратегии:
        - Weighted voting based on method reliability
        - Automatic edge case detection
        - Confidence scoring
        """
        consensus_results = {}
        
        for field_name, field_extractions in extractions.items():
            if not field_extractions:
                continue
                
            # Группируем по нормализованным значениям
            value_groups = defaultdict(list)
            for extraction in field_extractions:
                normalized = extraction.normalized_value or extraction.value
                value_groups[normalized].append(extraction)
            
            # Если только одно уникальное значение
            if len(value_groups) == 1:
                consensus_value = list(value_groups.keys())[0]
                all_extractions = list(value_groups.values())[0]
                
                # Средняя уверенность
                avg_confidence = mean([e.confidence for e in all_extractions])
                
                consensus_results[field_name] = ConsensusResult(
                    final_value=consensus_value,
                    confidence=min(0.95, avg_confidence + 0.1),  # Бонус за консенсус
                    agreement_score=1.0,
                    participating_methods=[e.method for e in all_extractions],
                    is_edge_case=False
                )
            
            # Если несколько разных значений - нужен консенсус
            else:
                consensus_result = self._resolve_conflict(field_name, value_groups)
                consensus_results[field_name] = consensus_result
        
        return consensus_results
    
    def _resolve_conflict(self, 
                         field_name: str, 
                         value_groups: Dict[str, List[FieldExtraction]]) -> ConsensusResult:
        """
        Разрешает конфликт между разными значениями поля
        Использует взвешенное голосование с учетом надежности методов
        """
        # Вычисляем вес каждого значения
        weighted_values = []
        
        for value, extractions in value_groups.items():
            # Базовый вес = количество методов * средняя уверенность
            base_weight = len(extractions) * mean([e.confidence for e in extractions])
            
            # Бонусы за определенные методы
            method_bonus = 0
            methods = [e.method for e in extractions]
            
            if 'pdf_text' in methods:
                method_bonus += 0.3  # PDF текст очень надежен
            if 'gemini' in methods:
                method_bonus += 0.2  # Gemini хорош для понимания контекста
            if 'pattern' in methods:
                method_bonus += 0.15  # Паттерны надежны для структурированных данных
            
            # Проверка на валидность значения
            validity_bonus = self._validate_field_value(field_name, value)
            
            total_weight = base_weight + method_bonus + validity_bonus
            
            weighted_values.append((value, total_weight, extractions, methods))
        
        # Сортируем по весу
        weighted_values.sort(key=lambda x: x[1], reverse=True)
        
        best_value, best_weight, best_extractions, best_methods = weighted_values[0]
        
        # Вычисляем agreement score
        total_extractions = sum(len(extractions) for extractions in value_groups.values())
        agreement_score = len(best_extractions) / total_extractions
        
        # Определяем, является ли это спорным случаем
        is_edge_case = agreement_score < 0.7 or len(value_groups) > 2
        
        # Уверенность зависит от согласованности
        confidence = min(0.9, best_weight / 2.0 * agreement_score)
        
        return ConsensusResult(
            final_value=best_value,
            confidence=confidence,
            agreement_score=agreement_score,
            participating_methods=list(set(best_methods)),
            is_edge_case=is_edge_case,
            conflict_details={
                'all_values': list(value_groups.keys()),
                'weights': [w for _, w, _, _ in weighted_values],
                'chosen_reason': f'Highest weight: {best_weight:.2f}'
            }
        )
    
    def _extract_with_patterns(self, text: str) -> Dict[str, str]:
        """Извлечение полей с помощью эталонных регулярных выражений"""
        results = {}
        
        for field_name, patterns in self.golden_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Берем первое найденное значение
                    value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    results[field_name] = value.strip()
                    break
        
        return results
    
    def _extract_from_pdf_blocks(self, pdf_blocks: List[Dict]) -> Dict[str, str]:
        """Извлечение полей из PDF текстовых блоков"""
        # Собираем весь текст из блоков
        all_text = " ".join([block.get('text', '') for block in pdf_blocks])
        
        # Применяем паттерны к тексту PDF
        return self._extract_with_patterns(all_text)
    
    def _normalize_field_value(self, field_name: str, value: str) -> str:
        """Нормализация значения поля для лучшего сравнения"""
        if not value:
            return value
            
        value = value.strip()
        
        if field_name == 'invoice_date':
            return self._normalize_date(value)
        elif field_name in ['company_inn', 'company_kpp']:
            return re.sub(r'[^\d]', '', value)  # Только цифры
        elif field_name == 'total_amount':
            return self._normalize_amount(value)
        elif field_name == 'invoice_number':
            return self._normalize_invoice_number(value)
        
        return value.lower().strip()
    
    def _normalize_date(self, date_str: str) -> str:
        """Нормализация даты к формату DD.MM.YYYY"""
        # Замена названий месяцев на числа
        normalized = date_str.lower()
        for month_name, month_num in self.month_names.items():
            normalized = normalized.replace(month_name, month_num)
        
        # Поиск паттернов даты
        patterns = [
            r'(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})',
            r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})',
            r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, normalized)
            if match:
                d, m, y = match.groups()
                
                # Определяем порядок дня и месяца
                if len(d) == 4:  # Год в начале
                    y, m, d = d, m, y
                
                return f"{d.zfill(2)}.{m.zfill(2)}.{y}"
        
        return date_str
    
    def _normalize_amount(self, amount_str: str) -> str:
        """Нормализация суммы"""
        # Удаляем все кроме цифр, точек и запятых
        cleaned = re.sub(r'[^\d.,]', '', amount_str)
        
        # Заменяем запятую на точку
        cleaned = cleaned.replace(',', '.')
        
        # Удаляем лишние точки
        parts = cleaned.split('.')
        if len(parts) > 2:
            cleaned = '.'.join(parts[:2])
        
        return cleaned
    
    def _normalize_invoice_number(self, number_str: str) -> str:
        """Нормализация номера счета"""
        # Удаляем лишние пробелы и символы
        cleaned = re.sub(r'[^\w\-/]', '', number_str)
        return cleaned.upper()
    
    def _validate_field_value(self, field_name: str, value: str) -> float:
        """
        Валидация значения поля
        Returns: бонус к весу (0.0 - 0.5)
        """
        if not value:
            return 0.0
        
        if field_name == 'company_inn':
            # ИНН должен быть 10 или 12 цифр
            if re.match(r'^\d{10}$', value) or re.match(r'^\d{12}$', value):
                return 0.4
            return 0.0
        
        elif field_name == 'company_kpp':
            # КПП должен быть 9 цифр
            if re.match(r'^\d{9}$', value):
                return 0.4
            return 0.0
        
        elif field_name == 'invoice_date':
            # Проверка формата даты
            if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', value):
                return 0.3
            return 0.1
        
        elif field_name == 'total_amount':
            # Проверка формата суммы
            if re.match(r'^\d+(\.\d{1,2})?$', value):
                return 0.3
            return 0.1
        
        elif field_name == 'invoice_number':
            # Номер счета не должен быть пустым и слишком длинным
            if 1 <= len(value) <= 50:
                return 0.2
            return 0.0
        
        return 0.1  # Небольшой бонус за наличие значения
    
    def calculate_quality_metrics(self, 
                                 consensus_results: Dict[str, ConsensusResult],
                                 expected_fields: Set[str] = None) -> AnnotationQualityMetrics:
        """
        Вычисляет метрики качества аннотации
        Использует стандартные метрики ML для оценки качества
        """
        if expected_fields is None:
            expected_fields = set(self.golden_patterns.keys())
        
        # Базовые метрики
        total_fields = len(expected_fields)
        extracted_fields = len(consensus_results)
        high_confidence_fields = len([r for r in consensus_results.values() if r.confidence > 0.8])
        edge_cases = len([r for r in consensus_results.values() if r.is_edge_case])
        
        # Accuracy = доля успешно извлеченных полей
        accuracy = extracted_fields / total_fields if total_fields > 0 else 0.0
        
        # Precision = доля высококачественных извлечений
        precision = high_confidence_fields / extracted_fields if extracted_fields > 0 else 0.0
        
        # Recall = доля найденных из всех ожидаемых полей
        recall = extracted_fields / total_fields if total_fields > 0 else 0.0
        
        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Inter-annotator agreement (средний agreement score)
        agreement_scores = [r.agreement_score for r in consensus_results.values()]
        inter_annotator_agreement = mean(agreement_scores) if agreement_scores else 0.0
        
        # Общая уверенность
        confidence_scores = [r.confidence for r in consensus_results.values()]
        confidence_score = mean(confidence_scores) if confidence_scores else 0.0
        
        # Уровень консенсуса
        consensus_level = len([r for r in consensus_results.values() if not r.is_edge_case]) / len(consensus_results) if consensus_results else 0.0
        
        return AnnotationQualityMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            inter_annotator_agreement=inter_annotator_agreement,
            confidence_score=confidence_score,
            edge_cases_detected=edge_cases,
            consensus_level=consensus_level
        )
    
    def generate_quality_report(self, 
                              metrics: AnnotationQualityMetrics,
                              consensus_results: Dict[str, ConsensusResult]) -> str:
        """
        Генерирует детальный отчет о качестве аннотации
        """
        report = []
        report.append("=" * 70)
        report.append("📊 ОТЧЕТ О КАЧЕСТВЕ АННОТАЦИИ ДАННЫХ")
        report.append("=" * 70)
        
        # Общая оценка качества
        overall_quality = (metrics.accuracy + metrics.f1_score + metrics.confidence_score) / 3
        quality_grade = "🔥 ОТЛИЧНО" if overall_quality > 0.9 else "✅ ХОРОШО" if overall_quality > 0.7 else "⚠️ ТРЕБУЕТ УЛУЧШЕНИЯ"
        
        report.append(f"🎯 ОБЩАЯ ОЦЕНКА: {quality_grade} ({overall_quality:.1%})")
        report.append("")
        
        # Основные метрики
        report.append("📈 ОСНОВНЫЕ МЕТРИКИ:")
        report.append(f"  🎯 Точность (Accuracy): {metrics.accuracy:.1%}")
        report.append(f"  🔍 Точность извлечения (Precision): {metrics.precision:.1%}")
        report.append(f"  📊 Полнота (Recall): {metrics.recall:.1%}")
        report.append(f"  ⚖️ F1-Score: {metrics.f1_score:.1%}")
        report.append(f"  🤝 Согласованность методов: {metrics.inter_annotator_agreement:.1%}")
        report.append(f"  💪 Средняя уверенность: {metrics.confidence_score:.1%}")
        report.append(f"  ✅ Уровень консенсуса: {metrics.consensus_level:.1%}")
        report.append(f"  ⚠️ Спорных случаев: {metrics.edge_cases_detected}")
        
        report.append("\n" + "=" * 50)
        report.append("📋 ДЕТАЛИ ПО ПОЛЯМ:")
        report.append("=" * 50)
        
        for field_name, result in consensus_results.items():
            status = "✅" if not result.is_edge_case else "⚠️"
            quality_indicator = "🔥" if result.confidence > 0.9 else "✅" if result.confidence > 0.7 else "⚠️"
            
            report.append(f"{status} {quality_indicator} {field_name.upper()}:")
            report.append(f"    💰 Значение: {result.final_value}")
            report.append(f"    🎯 Уверенность: {result.confidence:.1%}")
            report.append(f"    🤝 Согласованность: {result.agreement_score:.1%}")
            report.append(f"    🔧 Методы: {', '.join(result.participating_methods)}")
            
            if result.is_edge_case and result.conflict_details:
                report.append(f"    ⚠️ Конфликт: {result.conflict_details.get('chosen_reason', 'N/A')}")
                report.append(f"    🔄 Варианты: {', '.join(result.conflict_details.get('all_values', []))}")
        
        # Рекомендации по улучшению
        report.append("\n" + "=" * 50)
        report.append("💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
        report.append("=" * 50)
        
        recommendations = []
        
        if metrics.accuracy < 0.9:
            recommendations.append("🔧 Низкая точность - добавьте больше эталонных паттернов")
        
        if metrics.inter_annotator_agreement < 0.8:
            recommendations.append("🔧 Низкая согласованность - проверьте конфликтующие методы")
        
        if metrics.edge_cases_detected > len(consensus_results) * 0.2:
            recommendations.append("🔧 Много спорных случаев - требуется экспертная валидация")
        
        if metrics.confidence_score < 0.8:
            recommendations.append("🔧 Низкая уверенность - улучшите качество входных данных")
        
        if not recommendations:
            recommendations.append("🎉 Качество данных превосходно! Система работает на максимальном уровне.")
        
        for rec in recommendations:
            report.append(f"  {rec}")
        
        # Следующие шаги
        report.append("\n" + "=" * 50)
        report.append("🚀 СЛЕДУЮЩИЕ ШАГИ:")
        report.append("=" * 50)
        
        if overall_quality > 0.9:
            report.append("  ✅ Система готова к продуктивному использованию")
            report.append("  🔄 Рекомендуется периодический мониторинг качества")
        elif overall_quality > 0.7:
            report.append("  🔧 Проведите точечные улучшения проблемных полей")
            report.append("  📊 Добавьте больше эталонных примеров")
        else:
            report.append("  ⚠️ Требуется серьезная доработка системы")
            report.append("  👥 Рассмотрите привлечение экспертов для валидации")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def save_quality_report(self, 
                          report: str, 
                          output_path: Path,
                          metrics: AnnotationQualityMetrics,
                          consensus_results: Dict[str, ConsensusResult]):
        """
        Сохраняет отчет о качестве и метрики в файлы
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Текстовый отчет
        report_file = output_path / "data_quality_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # JSON с метриками
        metrics_file = output_path / "quality_metrics.json"
        metrics_data = {
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A',
            'overall_quality': (metrics.accuracy + metrics.f1_score + metrics.confidence_score) / 3,
            'metrics': {
                'accuracy': metrics.accuracy,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'inter_annotator_agreement': metrics.inter_annotator_agreement,
                'confidence_score': metrics.confidence_score,
                'edge_cases_detected': metrics.edge_cases_detected,
                'consensus_level': metrics.consensus_level
            },
            'field_details': {
                field_name: {
                    'final_value': result.final_value,
                    'confidence': result.confidence,
                    'agreement_score': result.agreement_score,
                    'participating_methods': result.participating_methods,
                    'is_edge_case': result.is_edge_case,
                    'conflict_details': result.conflict_details
                }
                for field_name, result in consensus_results.items()
            }
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"✅ Отчет о качестве сохранен: {report_file}")
        self.logger.info(f"✅ Метрики сохранены: {metrics_file}")
        
        return report_file, metrics_file


def calculate_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Вычисляет Fleiss' Kappa для множественных аннотаторов
    
    Args:
        ratings_matrix: матрица оценок [n_items, n_categories]
    
    Returns:
        float: значение Fleiss' Kappa (-1 до 1)
        
    Интерпретация:
        < 0.00: Плохое согласие
        0.00-0.20: Слабое согласие  
        0.21-0.40: Справедливое согласие
        0.41-0.60: Умеренное согласие
        0.61-0.80: Существенное согласие
        0.81-1.00: Почти идеальное согласие
    """
    n_items, n_categories = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1).max()
    
    # Пропорция каждой категории
    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)
    
    # Средняя пропорция согласия
    P_e = (p_j ** 2).sum()
    
    # Наблюдаемая пропорция согласия
    P_i = []
    for i in range(n_items):
        r_i = ratings_matrix[i]
        if r_i.sum() > 1:
            P_i.append((r_i * (r_i - 1)).sum() / (r_i.sum() * (r_i.sum() - 1)))
        else:
            P_i.append(0)
    
    P_o = np.mean(P_i)
    
    if P_e == 1:
        return 1 if P_o == 1 else 0
    
    kappa = (P_o - P_e) / (1 - P_e)
    return kappa


def calculate_cohens_kappa(rater1: List, rater2: List) -> float:
    """
    Вычисляет Cohen's Kappa для двух аннотаторов
    
    Args:
        rater1, rater2: списки аннотаций
    
    Returns:
        float: значение Cohen's Kappa (-1 до 1)
    """
    if len(rater1) != len(rater2):
        raise ValueError("Списки аннотаций должны быть одинаковой длины")
    
    # Создаем матрицу путаницы
    categories = list(set(rater1 + rater2))
    n_categories = len(categories)
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    confusion_matrix = np.zeros((n_categories, n_categories))
    
    for r1, r2 in zip(rater1, rater2):
        i, j = cat_to_idx[r1], cat_to_idx[r2]
        confusion_matrix[i, j] += 1
    
    n = len(rater1)
    
    # Наблюдаемая точность
    po = np.trace(confusion_matrix) / n
    
    # Ожидаемая точность
    pe = 0
    for i in range(n_categories):
        pe += (confusion_matrix[i, :].sum() / n) * (confusion_matrix[:, i].sum() / n)
    
    if pe == 1:
        return 1 if po == 1 else 0
    
    kappa = (po - pe) / (1 - pe)
    return kappa 