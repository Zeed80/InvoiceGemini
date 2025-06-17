"""
Pydantic models for data validation and serialization.
"""
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic.types import constr, conint
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseModel, Field, validator
    constr = str
    conint = int


class ModelType(str, Enum):
    """Supported model types."""
    LAYOUTLM = "layoutlm"
    DONUT = "donut"
    GEMINI = "gemini"
    CLOUD_LLM = "cloud_llm"
    LOCAL_LLM = "local_llm"


class FieldType(str, Enum):
    """Invoice field types."""
    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    EMAIL = "email"
    PHONE = "phone"


class ProcessingStatus(str, Enum):
    """Processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InvoiceField(BaseModel):
    """Single invoice field model."""
    key: str = Field(..., description="Field key identifier")
    name: str = Field(..., description="Field display name")
    value: Optional[Union[str, int, float, date]] = None
    type: FieldType = FieldType.TEXT
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    bbox: Optional[List[float]] = Field(None, min_items=4, max_items=4)
    page: Optional[int] = Field(None, ge=1)
    
    @validator('value', pre=True)
    def validate_value(cls, v, values):
        """Validate value based on field type."""
        if v is None:
            return v
            
        field_type = values.get('type', FieldType.TEXT)
        
        if field_type == FieldType.NUMBER:
            try:
                return float(v)
            except (ValueError, TypeError):
                raise ValueError(f"Invalid number value: {v}")
                
        elif field_type == FieldType.DATE:
            if isinstance(v, date):
                return v
            try:
                # Try common date formats
                for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"]:
                    try:
                        return datetime.strptime(str(v), fmt).date()
                    except ValueError:
                        continue
                raise ValueError(f"Invalid date value: {v}")
            except Exception:
                raise ValueError(f"Invalid date value: {v}")
                
        elif field_type == FieldType.CURRENCY:
            try:
                # Remove currency symbols and spaces
                cleaned = str(v).replace('$', '').replace('€', '').replace('₽', '').replace(' ', '').replace(',', '.')
                return Decimal(cleaned)
            except:
                raise ValueError(f"Invalid currency value: {v}")
                
        return str(v)


class InvoiceData(BaseModel):
    """Invoice data model."""
    # Required fields
    invoice_number: Optional[str] = Field(None, description="Invoice number")
    invoice_date: Optional[date] = Field(None, description="Invoice date")
    total_amount: Optional[Decimal] = Field(None, description="Total amount")
    
    # Company information
    supplier_name: Optional[str] = Field(None, description="Supplier/vendor name")
    supplier_address: Optional[str] = Field(None, description="Supplier address")
    supplier_tax_id: Optional[str] = Field(None, description="Supplier tax ID")
    
    customer_name: Optional[str] = Field(None, description="Customer name")
    customer_address: Optional[str] = Field(None, description="Customer address")
    customer_tax_id: Optional[str] = Field(None, description="Customer tax ID")
    
    # Financial details
    subtotal: Optional[Decimal] = Field(None, description="Subtotal amount")
    tax_amount: Optional[Decimal] = Field(None, description="Tax amount")
    tax_rate: Optional[float] = Field(None, ge=0, le=100, description="Tax rate percentage")
    discount_amount: Optional[Decimal] = Field(None, description="Discount amount")
    
    # Additional fields
    currency: Optional[str] = Field(None, max_length=3, description="Currency code")
    payment_terms: Optional[str] = Field(None, description="Payment terms")
    due_date: Optional[date] = Field(None, description="Payment due date")
    
    # Dynamic fields
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    model_used: Optional[ModelType] = None
    
    class Config:
        """Pydantic config."""
        json_encoders = {
            Decimal: str,
            date: lambda v: v.isoformat(),
            datetime: lambda v: v.isoformat(),
        }


class ExtractionResult(BaseModel):
    """Model extraction result."""
    status: ProcessingStatus
    invoice_data: Optional[InvoiceData] = None
    fields: List[InvoiceField] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Processing metadata
    file_path: Optional[Path] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = Field(None, ge=0)
    page_count: Optional[int] = Field(None, ge=1)
    
    # Timing information
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: Optional[float] = None
    
    # Model information
    model_type: Optional[ModelType] = None
    model_version: Optional[str] = None
    
    @root_validator
    def calculate_processing_time(cls, values):
        """Calculate processing time from start and end times."""
        start = values.get('start_time')
        end = values.get('end_time')
        
        if start and end and not values.get('processing_time'):
            values['processing_time'] = (end - start).total_seconds()
            
        return values


class BatchProcessingResult(BaseModel):
    """Batch processing result for multiple files."""
    total_files: int = Field(..., ge=0)
    processed_files: int = Field(0, ge=0)
    successful_files: int = Field(0, ge=0)
    failed_files: int = Field(0, ge=0)
    
    results: List[ExtractionResult] = Field(default_factory=list)
    
    # Timing
    start_time: datetime
    end_time: Optional[datetime] = None
    total_processing_time: Optional[float] = None
    
    # Statistics
    average_processing_time: Optional[float] = None
    average_confidence: Optional[float] = None
    
    @root_validator
    def calculate_statistics(cls, values):
        """Calculate batch statistics."""
        results = values.get('results', [])
        
        if results:
            # Count files
            values['processed_files'] = len(results)
            values['successful_files'] = sum(1 for r in results if r.status == ProcessingStatus.COMPLETED)
            values['failed_files'] = sum(1 for r in results if r.status == ProcessingStatus.FAILED)
            
            # Calculate averages
            processing_times = [r.processing_time for r in results if r.processing_time]
            if processing_times:
                values['average_processing_time'] = sum(processing_times) / len(processing_times)
                
            confidence_scores = [
                r.invoice_data.confidence_score 
                for r in results 
                if r.invoice_data and r.invoice_data.confidence_score
            ]
            if confidence_scores:
                values['average_confidence'] = sum(confidence_scores) / len(confidence_scores)
                
        return values


class ModelConfig(BaseModel):
    """Model configuration."""
    model_type: ModelType
    model_name: str
    model_version: Optional[str] = None
    
    # Resource requirements
    min_memory_mb: int = Field(1024, ge=0)
    min_gpu_memory_mb: Optional[int] = Field(None, ge=0)
    requires_gpu: bool = False
    
    # API configuration
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    timeout: int = Field(30, ge=1)
    max_retries: int = Field(3, ge=0)
    
    # Model parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic config."""
        extra = "allow"


class PluginConfig(BaseModel):
    """Plugin configuration."""
    plugin_id: str
    plugin_name: str
    plugin_type: str
    enabled: bool = True
    
    # Plugin settings
    settings: Dict[str, Any] = Field(default_factory=dict)
    
    # Dependencies
    required_packages: List[str] = Field(default_factory=list)
    min_version: Optional[str] = None
    max_version: Optional[str] = None


class ExportTemplate(BaseModel):
    """Export template configuration."""
    template_id: str
    template_name: str
    format: str = Field(..., regex="^(json|csv|excel|xml|pdf)$")
    
    # Field mapping
    field_mapping: Dict[str, str] = Field(default_factory=dict)
    include_fields: Optional[List[str]] = None
    exclude_fields: Optional[List[str]] = None
    
    # Formatting options
    date_format: str = "%Y-%m-%d"
    number_format: str = "{:.2f}"
    currency_symbol: Optional[str] = None
    
    # Template content (for custom formats)
    template_content: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        extra = "allow"


def validate_file_path(path: Union[str, Path]) -> Path:
    """Validate and sanitize file path."""
    path = Path(path)
    
    # Check for path traversal
    try:
        path.resolve()
    except Exception:
        raise ValueError(f"Invalid path: {path}")
        
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    # Check if it's a file
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
        
    # Check file extension
    allowed_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {path.suffix}")
        
    return path 