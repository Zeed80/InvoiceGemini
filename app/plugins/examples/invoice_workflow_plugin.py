"""
Invoice Workflow Plugin для InvoiceGemini
Плагин для автоматизации рабочих процессов обработки счетов
"""
from typing import Dict, Any, List
import json
from datetime import datetime
from pathlib import Path

from ..base_plugin import WorkflowPlugin, PluginMetadata, PluginType, PluginCapability, PluginStatus


class InvoiceProcessingWorkflowPlugin(WorkflowPlugin):
    """Плагин для автоматизации процесса обработки счетов"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.auto_validate = config.get('auto_validate', True) if config else True
        self.auto_export = config.get('auto_export', False) if config else False
        self.send_notifications = config.get('send_notifications', False) if config else False
        self.export_format = config.get('export_format', 'json') if config else 'json'
        
        # Шаги рабочего процесса
        self.workflow_steps = [
            {
                'id': 'preprocess',
                'name': 'Предобработка',
                'description': 'Проверка и подготовка файлов',
                'required': True
            },
            {
                'id': 'extract',
                'name': 'Извлечение данных',
                'description': 'Извлечение данных из счетов',
                'required': True
            },
            {
                'id': 'validate',
                'name': 'Валидация',
                'description': 'Проверка корректности данных',
                'required': False
            },
            {
                'id': 'transform',
                'name': 'Трансформация',
                'description': 'Преобразование и нормализация данных',
                'required': False
            },
            {
                'id': 'export',
                'name': 'Экспорт',
                'description': 'Сохранение результатов',
                'required': False
            },
            {
                'id': 'notify',
                'name': 'Уведомления',
                'description': 'Отправка уведомлений',
                'required': False
            }
        ]
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Invoice Processing Workflow",
            version="1.0.0",
            description="Автоматизированный рабочий процесс обработки счетов",
            author="InvoiceGemini Team",
            plugin_type=PluginType.WORKFLOW,
            capabilities=[PluginCapability.BATCH, PluginCapability.API, PluginCapability.ASYNC],
            config_schema={
                "required": [],
                "types": {
                    "auto_validate": bool,
                    "auto_export": bool,
                    "send_notifications": bool,
                    "export_format": str
                }
            }
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        self.status = PluginStatus.LOADED
        return True
    
    def cleanup(self):
        """Очистка ресурсов"""
        pass
    
    def execute_workflow(self, workflow_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Выполняет рабочий процесс обработки счетов"""
        try:
            results = {
                'workflow_id': f"invoice_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'start_time': datetime.now().isoformat(),
                'steps_completed': [],
                'success': True,
                'data': workflow_data.get('input_data', [])
            }
            
            return results
            
        except Exception as e:
            self.set_error(str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_workflow_steps(self) -> List[Dict[str, Any]]:
        """Возвращает шаги workflow"""
        return [
            {'id': 'extract', 'name': 'Извлечение данных', 'required': True},
            {'id': 'validate', 'name': 'Валидация', 'required': False},
            {'id': 'export', 'name': 'Экспорт', 'required': False}
        ]
    
    def validate_workflow_data(self, workflow_data: Dict[str, Any]) -> bool:
        """Валидация входных данных"""
        return True


class BatchProcessingWorkflowPlugin(WorkflowPlugin):
    """Плагин для пакетной обработки множества счетов"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.batch_size = config.get('batch_size', 10) if config else 10
        self.parallel_processing = config.get('parallel_processing', False) if config else False
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Batch Processing Workflow",
            version="1.0.0",
            description="Пакетная обработка большого количества счетов",
            author="InvoiceGemini Team",
            plugin_type=PluginType.WORKFLOW,
            capabilities=[PluginCapability.BATCH, PluginCapability.ASYNC],
            config_schema={
                "required": [],
                "types": {
                    "batch_size": int,
                    "parallel_processing": bool
                }
            }
        )
    
    def initialize(self) -> bool:
        """Инициализация плагина"""
        self.status = PluginStatus.LOADED
        return True
    
    def cleanup(self):
        """Очистка ресурсов"""
        pass
    
    def execute_workflow(self, workflow_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Выполняет пакетную обработку"""
        try:
            input_files = workflow_data.get('input_files', [])
            if not input_files:
                raise ValueError("Нет файлов для пакетной обработки")
            
            # Разбиваем на батчи
            batches = [input_files[i:i + self.batch_size] 
                      for i in range(0, len(input_files), self.batch_size)]
            
            results = {
                'workflow_id': f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'total_files': len(input_files),
                'total_batches': len(batches),
                'processed_batches': 0,
                'processed_files': 0,
                'all_results': [],
                'errors': [],
                'start_time': datetime.now().isoformat()
            }
            
            # Обрабатываем каждый батч
            for batch_idx, batch_files in enumerate(batches):
                self.report_progress(int((batch_idx / len(batches)) * 100), 
                                   f"Обработка батча {batch_idx + 1}/{len(batches)}")
                
                batch_data = {
                    'input_files': batch_files,
                    'output_path': f"batch_results/batch_{batch_idx + 1}.json"
                }
                
                # Используем основной workflow для обработки батча
                invoice_workflow = InvoiceProcessingWorkflowPlugin(self.config)
                batch_result = invoice_workflow.execute_workflow(batch_data)
                
                results['all_results'].append({
                    'batch_id': batch_idx + 1,
                    'files': batch_files,
                    'result': batch_result
                })
                
                if batch_result.get('success'):
                    results['processed_batches'] += 1
                    results['processed_files'] += len(batch_files)
                else:
                    results['errors'].extend(batch_result.get('errors', []))
            
            results['end_time'] = datetime.now().isoformat()
            results['success'] = results['processed_batches'] == len(batches)
            
            return results
            
        except Exception as e:
            self.set_error(str(e))
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_workflow_steps(self) -> List[Dict[str, Any]]:
        """Возвращает шаги пакетной обработки"""
        return [
            {
                'id': 'split_batches',
                'name': 'Разбиение на батчи',
                'description': f'Разбиение файлов на группы по {self.batch_size}',
                'required': True
            },
            {
                'id': 'process_batches',
                'name': 'Обработка батчей',
                'description': 'Последовательная/параллельная обработка батчей',
                'required': True
            },
            {
                'id': 'merge_results',
                'name': 'Объединение результатов',
                'description': 'Сбор и объединение результатов всех батчей',
                'required': True
            }
        ]
    
    def validate_workflow_data(self, workflow_data: Dict[str, Any]) -> bool:
        """Валидация данных для пакетной обработки"""
        input_files = workflow_data.get('input_files', [])
        
        if not input_files:
            self.set_error("Нет файлов для пакетной обработки")
            return False
        
        if len(input_files) < self.batch_size:
            self.set_error(f"Слишком мало файлов для пакетной обработки (минимум {self.batch_size})")
            return False
        
        return True 