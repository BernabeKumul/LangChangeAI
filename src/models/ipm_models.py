"""
Modelos Pydantic para auditorías IPM
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class FileSearch(BaseModel):
    """Modelo para archivos analizados en la auditoría"""
    FileName: str = Field(description="Nombre del archivo analizado")
    DocumentID: str = Field(description="ID único del documento")


class IPMAuditResponse(BaseModel):
    """Response model for IPM Compliance Audit"""
    ComplianceLevel: int = Field(description="Always returns the value 2", default=2)
    Comments: str = Field(description="Multi-paragraph compliance analysis")
    FilesSearch: List[FileSearch] = Field(description="List of files with FileName and DocumentID", default=[])


class DocumentInfo(BaseModel):
    """Información del documento para procesamiento"""
    filename: str
    doc_id: str
    content: str
    original_tokens: int = 0
    processed_tokens: int = 0 