"""
Unit tests for RAG-Powered Clinical Guidelines Module.
Tests chunking, vector indexing, citation retrieval, and honest insufficiency handling.
"""
import pytest
from modules.rag.retrieval import retrieval_service
from modules.rag.generation import guideline_service
from modules.rag.schemas import GuidelineAnswer

def test_rag_retrieval_success():
    citations = retrieval_service.retrieve("What is recommended for chest pain evaluation?", top_k=2)
    assert len(citations) > 0
    top_c = citations[0]
    assert top_c.page_number is not None
    assert top_c.similarity_score > 0.0
    assert len(top_c.snippet) > 10

def test_rag_generation_grounded():
    answer = guideline_service.answer_query("What are the treatment options for HFrEF heart failure?", top_k=2)
    assert isinstance(answer, GuidelineAnswer)
    assert answer.evidence_sufficient is True
    assert len(answer.citations) > 0

def test_rag_irrelevant_query_insufficient_evidence():
    # Irrelevant query completely absent from cardiology/pneumonia guidelines
    irrelevant_query = "What is the recommended recipe for baking sourdough bread in high altitude?"
    answer = guideline_service.answer_query(irrelevant_query, threshold=0.60)
    assert answer.evidence_sufficient is False
    assert "Insufficient evidence" in answer.answer
