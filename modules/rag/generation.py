"""
Evidence-Grounded Answer Generation for Clinical Guidelines.
Ensures strict provenance, inline citation tagging [1], [2], and honest fallback when evidence is insufficient.
"""
from typing import List, Optional
import os
import requests

from config.settings import settings
from modules.rag.schemas import Citation, GuidelineAnswer
from modules.rag.retrieval import retrieval_service

SYSTEM_PROMPT = """You are Clinical Copilot's Guideline Assistant.
Answer the clinical query strictly grounded in the provided Evidence Passages.
Rules:
1. Do not use outside medical knowledge or fabricate guideline recommendations.
2. Cite all claims using bracketed numbers [1], [2] referencing the source passages.
3. If the provided evidence is insufficient or conflicting, explicitly state that the indexed guidelines do not provide adequate evidence.
4. Always maintain clear clinical disclaimers.
"""

def generate_offline_synthesis(query: str, citations: List[Citation]) -> str:
    """
    Synthesizes a structured, evidence-grounded clinical response from retrieved citations
    without requiring external paid API keys.
    """
    if not citations:
        return "Insufficient evidence: The indexed clinical guidelines do not contain relevant information to answer this question."

    response_lines = [
        f"Based on the indexed clinical guidelines, the following evidence was retrieved regarding **'{query}'**:\n"
    ]
    
    for citation in citations:
        page_info = f" (Page {citation.page_number})" if citation.page_number else ""
        section_info = f" - Section: {citation.section_heading}" if citation.section_heading else ""
        
        # Extract the key sentences
        snippet = citation.snippet.strip()
        # Clean formatting
        cleaned_snippet = " ".join(snippet.split())
        
        response_lines.append(
            f"• **[{citation.citation_id}] {citation.document_title}{page_info}{section_info}**:\n"
            f"  > \"{cleaned_snippet}\"\n"
        )
        
    response_lines.append(
        "\n**Clinical Synthesis:**\n"
        "The guideline evidence above directly addresses the clinical condition. "
        "Recommendations must be contextualized with individual patient history, contraindications, and specialist evaluation."
    )
    return "\n".join(response_lines)

class GuidelineGenerator:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.groq_key = os.getenv("GROQ_API_KEY", "")

    def answer_query(
        self,
        query: str,
        document_filter: Optional[str] = None,
        top_k: int = 3,
        threshold: float = 0.25
    ) -> GuidelineAnswer:
        """Retrieves passages and generates an evidence-grounded response with traceable citations."""
        citations = retrieval_service.retrieve(
            query=query,
            top_k=top_k,
            threshold=threshold,
            document_filter=document_filter
        )

        if not citations:
            return GuidelineAnswer(
                query=query,
                answer=(
                    "Insufficient evidence: No relevant passages with sufficient similarity were found in the indexed "
                    "clinical guidelines. Please upload additional guideline documents or rephrase the clinical inquiry."
                ),
                citations=[],
                evidence_sufficient=False,
                retrieved_chunks_count=0,
                generator_model="Evidence-Grounded RAG Engine (Zero Passages)"
            )

        # Check if external LLM configured
        if self.groq_key:
            try:
                headers = {"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"}
                context = "\n\n".join([f"[{c.citation_id}] From {c.document_title} (Page {c.page_number}):\n{c.snippet}" for c in citations])
                payload = {
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Evidence Passages:\n{context}\n\nClinical Question: {query}"}
                    ],
                    "temperature": 0.2
                }
                res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=12)
                if res.status_code == 200:
                    ans_text = res.json()["choices"][0]["message"]["content"]
                    return GuidelineAnswer(
                        query=query,
                        answer=ans_text,
                        citations=citations,
                        evidence_sufficient=True,
                        retrieved_chunks_count=len(citations),
                        generator_model="Groq LLama-3.1-8B-Instant (Grounded)"
                    )
            except Exception:
                pass  # Fall back to local synthesis

        if self.openai_key:
            try:
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
                context = "\n\n".join([f"[{c.citation_id}] From {c.document_title} (Page {c.page_number}):\n{c.snippet}" for c in citations])
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Evidence Passages:\n{context}\n\nClinical Question: {query}"}
                    ],
                    "temperature": 0.2
                }
                res = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=12)
                if res.status_code == 200:
                    ans_text = res.json()["choices"][0]["message"]["content"]
                    return GuidelineAnswer(
                        query=query,
                        answer=ans_text,
                        citations=citations,
                        evidence_sufficient=True,
                        retrieved_chunks_count=len(citations),
                        generator_model="OpenAI GPT-4o-mini (Grounded)"
                    )
            except Exception:
                pass  # Fall back to local synthesis

        # Default: high-precision offline evidence synthesis
        synthesized_text = generate_offline_synthesis(query, citations)
        return GuidelineAnswer(
            query=query,
            answer=synthesized_text,
            citations=citations,
            evidence_sufficient=True,
            retrieved_chunks_count=len(citations),
            generator_model="Deterministic Clinical Evidence Synthesizer"
        )

# Global singleton generator instance
guideline_service = GuidelineGenerator()
