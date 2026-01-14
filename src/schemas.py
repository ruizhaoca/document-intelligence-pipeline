"""
Data schemas for document intelligence pipeline using Pydantic models.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Any
from enum import Enum
import uuid


class DocumentType(str, Enum):
    """Supported document types"""
    INVOICE = "invoice"
    CONTRACT = "contract"
    EMAIL = "email"
    MEETING_MINUTES = "meeting_minutes"
    UNKNOWN = "unknown"


class BaseDocument(BaseModel):
    """Base document schema - all documents inherit this"""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_type: DocumentType
    file_name: str
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    confidence_score: float = Field(ge=0.0, le=1.0)
    involved_parties: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None

    class Config:
        use_enum_values = True


class LineItem(BaseModel):
    """Line item for invoices"""
    description: str
    amount: float


class Invoice(BaseDocument):
    """Invoice document schema"""
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None  # Using string for flexibility with date parsing
    client_name: Optional[str] = None
    vendor_name: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    line_items: List[LineItem] = Field(default_factory=list)
    payment_method: Optional[str] = None


class Contract(BaseDocument):
    """Contract document schema"""
    contract_id: Optional[str] = None
    contract_date: Optional[str] = None
    parties: List[str] = Field(default_factory=list)
    contract_value: Optional[float] = None
    currency: Optional[str] = None
    effective_date: Optional[str] = None
    expiry_date: Optional[str] = None
    key_terms: Optional[str] = None
    contract_type: Optional[str] = None


class Email(BaseDocument):
    """Email document schema"""
    sender: Optional[str] = None
    recipients: List[str] = Field(default_factory=list)
    email_date: Optional[str] = None
    subject: Optional[str] = None
    key_points: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    """Action item for meeting minutes"""
    task: str
    assignee: Optional[str] = None
    deadline: Optional[str] = None


class MeetingMinutes(BaseDocument):
    """Meeting minutes document schema"""
    meeting_date: Optional[str] = None
    meeting_title: Optional[str] = None
    attendees: List[str] = Field(default_factory=list)
    agenda_items: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)
    next_meeting: Optional[str] = None


# Document type mapping for factory pattern
DOCUMENT_TYPE_MAP = {
    DocumentType.INVOICE: Invoice,
    DocumentType.CONTRACT: Contract,
    DocumentType.EMAIL: Email,
    DocumentType.MEETING_MINUTES: MeetingMinutes,
}


def create_document(doc_type: DocumentType, **kwargs) -> BaseDocument:
    """Factory function to create the appropriate document type"""
    document_class = DOCUMENT_TYPE_MAP.get(doc_type, BaseDocument)
    return document_class(document_type=doc_type, **kwargs)
