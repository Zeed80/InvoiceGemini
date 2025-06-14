#!/usr/bin/env python3
"""
Model Comparison Processor for InvoiceGemini
Runs multiple models on the same input and compares results.
"""

import os
import time
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from PyQt6.QtCore import QObject, pyqtSignal, QThread

from ..settings_manager import settings_manager
from .. import utils
from .. import config as app_config


class ModelComparisonResult:
    """Container for comparison results from a single model"""
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.results: Dict[str, Any] = {}
        self.processing_time: float = 0.0
        self.success: bool = False
        self.error_message: str = ""
        self.accuracy_metrics: Dict[str, float] = {}
        self.confidence_scores: Dict[str, float] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'results': self.results,
            'processing_time': self.processing_time,
            'success': self.success,
            'error_message': self.error_message,
            'accuracy_metrics': self.accuracy_metrics,
            'confidence_scores': self.confidence_scores
        }


class ModelComparisonProcessor(QObject):
    """
    Processor for comparing results from multiple models.
    Features:
    - Parallel execution of models
    - Result comparison and analysis
    - Accuracy metrics calculation
    - Confidence scoring
    """
    
    # Signals for progress reporting
    comparison_started = pyqtSignal()
    model_started = pyqtSignal(str)  # model_name
    model_completed = pyqtSignal(str, dict)  # model_name, results
    model_failed = pyqtSignal(str, str)  # model_name, error_message
    comparison_completed = pyqtSignal(dict)  # comprehensive comparison results
    progress_updated = pyqtSignal(int)  # progress percentage
    
    def __init__(self, model_manager=None, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.comparison_results: Dict[str, ModelComparisonResult] = {}
        self.is_running = False
        self.stop_requested = False
        
        # Comparison settings
        self.max_parallel_models = 3  # Maximum models to run in parallel
        self.timeout_per_model = 300  # 5 minutes timeout per model
        
        # Field weights for accuracy calculation
        self.field_weights = self._load_field_weights()
    
    def _load_field_weights(self) -> Dict[str, float]:
        """Load field importance weights from settings"""
        try:
            # Get table fields and assign weights based on field type
            table_fields = settings_manager.get_table_fields()
            weights = {}
            
            for field in table_fields:
                field_id = field.get("id", "")
                field_type = field.get("type", "text")
                
                # Assign weights based on field importance
                if "amount" in field_id.lower() or "sum" in field_id.lower():
                    weights[field_id] = 1.0  # Highest priority
                elif "date" in field_id.lower() or "number" in field_id.lower():
                    weights[field_id] = 0.9  # High priority
                elif "supplier" in field_id.lower() or "vendor" in field_id.lower():
                    weights[field_id] = 0.8  # Medium-high priority
                else:
                    weights[field_id] = 0.5  # Standard priority
            
            return weights
            
        except Exception as e:
            print(f"Error loading field weights: {e}")
            return {}
    
    def compare_models(self, file_path: str, models_to_compare: List[str], 
                      reference_results: Optional[Dict] = None) -> bool:
        """
        Start model comparison process.
        
        Args:
            file_path: Path to the file to process
            models_to_compare: List of model names/types to compare
            reference_results: Optional reference results for accuracy calculation
            
        Returns:
            bool: True if comparison started successfully
        """
        if self.is_running:
            return False
        
        try:
            self.is_running = True
            self.stop_requested = False
            self.comparison_results.clear()
            
            # Validate inputs
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not models_to_compare:
                raise ValueError("No models specified for comparison")
            
            # Start comparison in thread
            self.comparison_thread = ModelComparisonThread(
                file_path=file_path,
                models_to_compare=models_to_compare,
                reference_results=reference_results,
                model_manager=self.model_manager,
                field_weights=self.field_weights,
                max_parallel=self.max_parallel_models,
                timeout=self.timeout_per_model,
                parent=self
            )
            
            # Connect signals
            self.comparison_thread.model_started.connect(self.model_started)
            self.comparison_thread.model_completed.connect(self._on_model_completed)
            self.comparison_thread.model_failed.connect(self.model_failed)
            self.comparison_thread.comparison_completed.connect(self._on_comparison_completed)
            self.comparison_thread.progress_updated.connect(self.progress_updated)
            
            self.comparison_thread.start()
            self.comparison_started.emit()
            
            return True
            
        except Exception as e:
            self.is_running = False
            print(f"Error starting model comparison: {e}")
            return False
    
    def stop_comparison(self):
        """Stop the comparison process"""
        self.stop_requested = True
        if hasattr(self, 'comparison_thread') and self.comparison_thread.isRunning():
            self.comparison_thread.stop_requested = True
            self.comparison_thread.wait(5000)  # Wait up to 5 seconds
    
    def _on_model_completed(self, model_name: str, result_data: Dict):
        """Handle completion of a single model"""
        if model_name in self.comparison_results:
            self.comparison_results[model_name].results = result_data.get('results', {})
            self.comparison_results[model_name].processing_time = result_data.get('processing_time', 0.0)
            self.comparison_results[model_name].success = True
            self.comparison_results[model_name].confidence_scores = result_data.get('confidence_scores', {})
        
        self.model_completed.emit(model_name, result_data)
    
    def _on_comparison_completed(self, comprehensive_results: Dict):
        """Handle completion of all model comparisons"""
        self.is_running = False
        
        # Store results
        for model_name, result_data in comprehensive_results.get('model_results', {}).items():
            if model_name in self.comparison_results:
                self.comparison_results[model_name].accuracy_metrics = result_data.get('accuracy_metrics', {})
        
        self.comparison_completed.emit(comprehensive_results)
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Get a summary of comparison results"""
        if not self.comparison_results:
            return {}
        
        summary = {
            'total_models': len(self.comparison_results),
            'successful_models': sum(1 for r in self.comparison_results.values() if r.success),
            'failed_models': sum(1 for r in self.comparison_results.values() if not r.success),
            'average_processing_time': 0.0,
            'best_model': None,
            'accuracy_ranking': [],
            'field_agreement': {},
            'consensus_results': {}
        }
        
        # Calculate average processing time
        successful_results = [r for r in self.comparison_results.values() if r.success]
        if successful_results:
            summary['average_processing_time'] = sum(r.processing_time for r in successful_results) / len(successful_results)
        
        # Determine best model based on overall accuracy
        best_model = None
        best_accuracy = 0.0
        
        for model_name, result in self.comparison_results.items():
            if result.success and result.accuracy_metrics:
                overall_accuracy = result.accuracy_metrics.get('overall_accuracy', 0.0)
                if overall_accuracy > best_accuracy:
                    best_accuracy = overall_accuracy
                    best_model = model_name
        
        summary['best_model'] = best_model
        
        # Create accuracy ranking
        ranking = []
        for model_name, result in self.comparison_results.items():
            if result.success:
                accuracy = result.accuracy_metrics.get('overall_accuracy', 0.0)
                ranking.append((model_name, accuracy))
        
        ranking.sort(key=lambda x: x[1], reverse=True)
        summary['accuracy_ranking'] = ranking
        
        # Calculate field agreement (consensus)
        summary['field_agreement'] = self._calculate_field_agreement()
        summary['consensus_results'] = self._generate_consensus_results()
        
        return summary
    
    def _calculate_field_agreement(self) -> Dict[str, Dict[str, Any]]:
        """Calculate agreement between models for each field"""
        field_agreement = {}
        
        # Get all unique fields across all successful results
        all_fields = set()
        successful_results = {name: result for name, result in self.comparison_results.items() if result.success}
        
        for result in successful_results.values():
            all_fields.update(result.results.keys())
        
        # Calculate agreement for each field
        for field in all_fields:
            field_values = []
            model_names = []
            
            for model_name, result in successful_results.items():
                if field in result.results:
                    field_values.append(str(result.results[field]).strip().lower())
                    model_names.append(model_name)
            
            if len(field_values) > 1:
                # Calculate agreement percentage
                unique_values = list(set(field_values))
                most_common_value = max(unique_values, key=field_values.count)
                agreement_count = field_values.count(most_common_value)
                agreement_percentage = (agreement_count / len(field_values)) * 100
                
                field_agreement[field] = {
                    'agreement_percentage': agreement_percentage,
                    'total_models': len(field_values),
                    'agreeing_models': agreement_count,
                    'most_common_value': most_common_value,
                    'all_values': list(zip(model_names, field_values)),
                    'is_consensus': agreement_percentage >= 50.0
                }
        
        return field_agreement
    
    def _generate_consensus_results(self) -> Dict[str, Any]:
        """Generate consensus results based on model agreement"""
        consensus = {}
        field_agreement = self._calculate_field_agreement()
        
        for field, agreement_data in field_agreement.items():
            if agreement_data['is_consensus']:
                # Use the most common value as consensus
                consensus[field] = agreement_data['most_common_value']
            else:
                # No clear consensus - might need human review
                consensus[field] = f"[REVIEW NEEDED] Multiple values: {', '.join(set(v for _, v in agreement_data['all_values']))}"
        
        return consensus
    
    def export_comparison_report(self, file_path: str) -> bool:
        """Export detailed comparison report"""
        try:
            summary = self.get_comparison_summary()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'comparison_summary': summary,
                'detailed_results': {name: result.to_dict() for name, result in self.comparison_results.items()},
                'field_weights': self.field_weights
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting comparison report: {e}")
            return False


class ModelComparisonThread(QThread):
    """Thread for running model comparison in background"""
    
    model_started = pyqtSignal(str)
    model_completed = pyqtSignal(str, dict)
    model_failed = pyqtSignal(str, str)
    comparison_completed = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, file_path: str, models_to_compare: List[str], 
                 reference_results: Optional[Dict], model_manager,
                 field_weights: Dict[str, float], max_parallel: int, 
                 timeout: int, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.models_to_compare = models_to_compare
        self.reference_results = reference_results
        self.model_manager = model_manager
        self.field_weights = field_weights
        self.max_parallel = max_parallel
        self.timeout = timeout
        self.stop_requested = False
        
    def run(self):
        """Main comparison execution"""
        try:
            results = {}
            total_models = len(self.models_to_compare)
            completed_models = 0
            
            # Run models in parallel batches
            with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                # Submit all model tasks
                future_to_model = {}
                for model_name in self.models_to_compare:
                    if self.stop_requested:
                        break
                    
                    future = executor.submit(self._run_single_model, model_name)
                    future_to_model[future] = model_name
                
                # Collect results as they complete
                for future in as_completed(future_to_model, timeout=self.timeout * total_models):
                    if self.stop_requested:
                        break
                    
                    model_name = future_to_model[future]
                    
                    try:
                        result_data = future.result(timeout=self.timeout)
                        results[model_name] = result_data
                        self.model_completed.emit(model_name, result_data)
                        
                    except Exception as e:
                        error_msg = str(e)
                        results[model_name] = {
                            'success': False,
                            'error': error_msg,
                            'processing_time': 0.0
                        }
                        self.model_failed.emit(model_name, error_msg)
                    
                    completed_models += 1
                    progress = int((completed_models / total_models) * 100)
                    self.progress_updated.emit(progress)
            
            # Calculate comprehensive comparison results
            comprehensive_results = self._calculate_comprehensive_results(results)
            self.comparison_completed.emit(comprehensive_results)
            
        except Exception as e:
            print(f"Error in model comparison thread: {e}")
            self.comparison_completed.emit({'error': str(e)})
    
    def _run_single_model(self, model_name: str) -> Dict[str, Any]:
        """Run a single model and return results"""
        start_time = time.time()
        self.model_started.emit(model_name)
        
        try:
            # Get model processor
            processor = self.model_manager.get_model(model_name.lower())
            if not processor:
                raise RuntimeError(f"Model {model_name} not available")
            
            # Process the file
            results = processor.extract_data(self.file_path)
            
            processing_time = time.time() - start_time
            
            # Calculate confidence scores if available
            confidence_scores = getattr(processor, 'confidence_scores', {})
            
            return {
                'success': True,
                'results': results,
                'processing_time': processing_time,
                'confidence_scores': confidence_scores,
                'model_info': {
                    'name': model_name,
                    'type': type(processor).__name__
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'results': {},
                'confidence_scores': {}
            }
    
    def _calculate_comprehensive_results(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate comprehensive comparison results and metrics"""
        comprehensive = {
            'model_results': {},
            'comparison_metrics': {},
            'field_analysis': {},
            'recommendations': []
        }
        
        # Process each model's results
        for model_name, result_data in model_results.items():
            if not result_data.get('success', False):
                comprehensive['model_results'][model_name] = result_data
                continue
            
            # Calculate accuracy metrics if reference results are available
            accuracy_metrics = {}
            if self.reference_results:
                accuracy_metrics = self._calculate_accuracy_metrics(
                    result_data['results'], 
                    self.reference_results
                )
            
            comprehensive['model_results'][model_name] = {
                **result_data,
                'accuracy_metrics': accuracy_metrics
            }
        
        # Calculate overall comparison metrics
        comprehensive['comparison_metrics'] = self._calculate_comparison_metrics(model_results)
        
        # Analyze field-level performance
        comprehensive['field_analysis'] = self._analyze_field_performance(model_results)
        
        # Generate recommendations
        comprehensive['recommendations'] = self._generate_recommendations(model_results)
        
        return comprehensive
    
    def _calculate_accuracy_metrics(self, model_results: Dict, reference_results: Dict) -> Dict[str, float]:
        """Calculate accuracy metrics against reference results"""
        metrics = {
            'field_accuracy': 0.0,
            'value_similarity': 0.0,
            'completeness': 0.0,
            'overall_accuracy': 0.0
        }
        
        try:
            # Field accuracy - percentage of fields that match exactly
            matching_fields = 0
            total_fields = len(reference_results)
            
            for field, ref_value in reference_results.items():
                model_value = model_results.get(field, "")
                if str(ref_value).strip().lower() == str(model_value).strip().lower():
                    matching_fields += 1
            
            metrics['field_accuracy'] = (matching_fields / total_fields) * 100 if total_fields > 0 else 0
            
            # Completeness - percentage of reference fields that have values in model results
            filled_fields = sum(1 for field in reference_results.keys() if model_results.get(field, "").strip())
            metrics['completeness'] = (filled_fields / total_fields) * 100 if total_fields > 0 else 0
            
            # Value similarity - weighted average based on field importance
            weighted_similarity = 0.0
            total_weight = 0.0
            
            for field, ref_value in reference_results.items():
                model_value = model_results.get(field, "")
                field_weight = self.field_weights.get(field, 0.5)
                
                # Simple similarity calculation
                similarity = 1.0 if str(ref_value).strip().lower() == str(model_value).strip().lower() else 0.0
                
                weighted_similarity += similarity * field_weight
                total_weight += field_weight
            
            metrics['value_similarity'] = (weighted_similarity / total_weight) * 100 if total_weight > 0 else 0
            
            # Overall accuracy - combined metric
            metrics['overall_accuracy'] = (
                metrics['field_accuracy'] * 0.4 +
                metrics['value_similarity'] * 0.4 +
                metrics['completeness'] * 0.2
            )
            
        except Exception as e:
            print(f"Error calculating accuracy metrics: {e}")
        
        return metrics
    
    def _calculate_comparison_metrics(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall comparison metrics"""
        successful_models = [r for r in model_results.values() if r.get('success', False)]
        
        if not successful_models:
            return {'error': 'No successful model results'}
        
        return {
            'total_models': len(model_results),
            'successful_models': len(successful_models),
            'average_processing_time': sum(r['processing_time'] for r in successful_models) / len(successful_models),
            'fastest_model': min(successful_models, key=lambda x: x['processing_time'])['model_info']['name'],
            'success_rate': (len(successful_models) / len(model_results)) * 100
        }
    
    def _analyze_field_performance(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze performance at the field level"""
        field_analysis = {}
        
        # Get all unique fields
        all_fields = set()
        for result in model_results.values():
            if result.get('success', False):
                all_fields.update(result['results'].keys())
        
        # Analyze each field
        for field in all_fields:
            field_data = {
                'extraction_rate': 0.0,  # Percentage of models that extracted this field
                'value_consistency': 0.0,  # How consistent the values are across models
                'average_confidence': 0.0,  # Average confidence scores
                'values_by_model': {}
            }
            
            successful_extractions = 0
            field_values = []
            confidence_scores = []
            
            for model_name, result in model_results.items():
                if result.get('success', False):
                    field_value = result['results'].get(field, "")
                    field_data['values_by_model'][model_name] = field_value
                    
                    if field_value.strip():
                        successful_extractions += 1
                        field_values.append(field_value.strip().lower())
                    
                    # Get confidence score if available
                    confidence = result.get('confidence_scores', {}).get(field, 0.0)
                    if confidence > 0:
                        confidence_scores.append(confidence)
            
            # Calculate metrics
            total_models = len([r for r in model_results.values() if r.get('success', False)])
            field_data['extraction_rate'] = (successful_extractions / total_models) * 100 if total_models > 0 else 0
            
            # Value consistency (based on agreement)
            if field_values:
                unique_values = list(set(field_values))
                most_common_count = max(field_values.count(val) for val in unique_values)
                field_data['value_consistency'] = (most_common_count / len(field_values)) * 100
            
            # Average confidence
            field_data['average_confidence'] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            field_analysis[field] = field_data
        
        return field_analysis
    
    def _generate_recommendations(self, model_results: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on comparison results"""
        recommendations = []
        
        successful_models = [(name, r) for name, r in model_results.items() if r.get('success', False)]
        
        if not successful_models:
            recommendations.append("❌ Все модели завершились с ошибками. Проверьте входные данные и конфигурацию.")
            return recommendations
        
        # Performance recommendations
        if len(successful_models) > 1:
            fastest = min(successful_models, key=lambda x: x[1]['processing_time'])
            slowest = max(successful_models, key=lambda x: x[1]['processing_time'])
            
            if fastest[1]['processing_time'] * 2 < slowest[1]['processing_time']:
                recommendations.append(f"⚡ Модель {fastest[0]} значительно быстрее других ({fastest[1]['processing_time']:.1f}с vs {slowest[1]['processing_time']:.1f}с)")
        
        # Accuracy recommendations (if reference available)
        if self.reference_results:
            best_accuracy = 0
            best_model = None
            
            for model_name, result in successful_models:
                accuracy = result.get('accuracy_metrics', {}).get('overall_accuracy', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
            
            if best_model and best_accuracy > 80:
                recommendations.append(f"🎯 Модель {best_model} показала лучшую точность ({best_accuracy:.1f}%)")
            elif best_accuracy < 60:
                recommendations.append("⚠️ Все модели показали низкую точность. Рассмотрите улучшение качества входных данных.")
        
        # Consensus recommendations
        # Calculate field agreement for recommendations
        field_agreement = {}
        all_fields = set()
        
        for _, result in successful_models:
            all_fields.update(result['results'].keys())
        
        low_agreement_fields = []
        for field in all_fields:
            field_values = []
            for _, result in successful_models:
                if field in result['results']:
                    field_values.append(str(result['results'][field]).strip().lower())
            
            if len(field_values) > 1:
                unique_values = list(set(field_values))
                agreement = (field_values.count(max(unique_values, key=field_values.count)) / len(field_values)) * 100
                if agreement < 50:
                    low_agreement_fields.append(field)
        
        if low_agreement_fields:
            recommendations.append(f"🔍 Поля требуют проверки из-за низкого согласия между моделями: {', '.join(low_agreement_fields[:3])}{'...' if len(low_agreement_fields) > 3 else ''}")
        
        return recommendations 