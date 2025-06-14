"""
Базовый абстрактный класс для всех процессоров в приложении.
"""
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    """
    Базовый абстрактный класс для процессоров моделей.
    Определяет общий интерфейс для всех процессоров.
    """
    
    @abstractmethod
    def process_image(self, image_path, ocr_lang=None, custom_prompt=None):
        """
        Обрабатывает изображение и извлекает из него информацию.
        
        Args:
            image_path (str): Путь к файлу изображения
            ocr_lang (str, optional): Язык OCR для моделей, использующих Tesseract
            custom_prompt (str, optional): Пользовательский промпт для моделей, поддерживающих его
            
        Returns:
            dict: Словарь с извлеченными данными
        """
        pass
    
    def get_full_prompt(self):
        """
        Возвращает полный запрос, который отправляется модели.
        
        Returns:
            str: Полный запрос к модели
        """
        return "Базовый метод. Переопределите в дочернем классе."
    
    # 🆕 НОВЫЕ МЕТОДЫ для поддержки плагинной системы (с дефолтными реализациями для совместимости)
    def supports_training(self) -> bool:
        """
        Поддерживает ли процессор обучение на пользовательских данных.
        
        Returns:
            bool: True если поддерживает обучение, False если нет
        """
        return False  # По умолчанию не поддерживает для обратной совместимости
    
    def get_trainer_class(self):
        """
        Возвращает класс тренера для этого процессора.
        
        Returns:
            class: Класс тренера или None если обучение не поддерживается
        """
        return None
    
    def get_model_type(self) -> str:
        """
        Возвращает тип модели для идентификации.
        
        Returns:
            str: Тип модели ('layoutlm', 'donut', 'gemini', 'llm_plugin')
        """
        return "unknown"
    
    def get_model_info(self) -> dict:
        """
        Возвращает информацию о модели.
        
        Returns:
            dict: Информация о модели
        """
        return {
            "type": self.get_model_type(),
            "supports_training": self.supports_training(),
            "name": getattr(self, 'model_id', 'Unknown')
        } 