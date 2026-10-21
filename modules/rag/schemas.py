"""
Data schemas for RAG-Powered Clinical Guidelines Module.
Enforces traceable citations, page references, and evidence sufficiency validation.
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_title: str = Field(..., description="Document filename or clinical guideline title")
    page_number: Optional[int] = Field(None, description="Page number within source document")
    section_heading: Optional[str] = Field(None, description="Clinical section heading (e.g. Diagnostic Workup, Pharmacotherapy)")
    text: str = Field(..., description="Cleaned clinical text passage")
    token_count: Optional[int] = Field(None, description="Approximate token count")

class Citation(BaseModel):
    citation_id: int = Field(..., description="Numeric citation marker [1], [2]")
    document_title: str = Field(..., description="Source guideline document")
    page_number: Optional[int] = Field(None, description="Exact page reference")
    section_heading: Optional[str] = Field(None, description="Guideline section")
    snippet: str = Field(..., description="Exact or summarized supporting text from document")
    similarity_score: float = Field(..., description="Semantic vector similarity score (0.0 to 1.0)")

class GuidelineAnswer(BaseModel):
    query: str = Field(..., description="Clinician's question")
    answer: str = Field(..., description="Synthesized clinical answer strictly grounded in retrieved passages")
    citations: List[Citation] = Field(default_factory=list, description="Traceable bibliographic citations")
    evidence_sufficient: bool = Field(True, description="False if retrieved passages lack evidence to answer safely")
    retrieved_chunks_count: int = Field(0, description="Total chunks retrieved above similarity threshold")
    disclaimer: str = Field(
        "Clinical guidance derived solely from indexed guideline documentation. Clinicians must apply independent judgment.",
        description="Clinical safety disclaimer"
    )
    generator_model: str = Field("Evidence-Grounded RAG Engine", description="LLM or generation component")
