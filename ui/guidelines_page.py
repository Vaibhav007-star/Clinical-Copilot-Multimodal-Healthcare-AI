"""
Clinical Guidelines Assistant UI (Page 4).
Interactive RAG interface with document indexing, filtering, expandable citations, and evidence sufficiency validation.
"""
from pathlib import Path
import streamlit as st
import pandas as pd

from config.settings import settings
from modules.rag.retrieval import retrieval_service
from modules.rag.generation import guideline_service
from modules.rag.schemas import GuidelineAnswer

SAMPLE_QUESTIONS = [
    "Select an example clinical question...",
    "What are the indications for chest radiography in acute chest pain?",
    "What is the foundational quadruple GDMT therapy recommended for HFrEF?",
    "What are the empirical antibiotic regimens for outpatient community-acquired pneumonia?",
    "How does the HEART score stratify 30-day major adverse cardiac event risk?",
    "What are the diagnostic criteria and radiographic signs of pneumonia?"
]

def render_guidelines_page():
    st.title("📚 RAG-Powered Clinical Guidelines Assistant")
    st.markdown(
        "Ask clinical questions grounded strictly in indexed clinical practice guidelines "
        "(AHA/ACC Chest Pain, AHA/ACC Heart Failure, ATS/IDSA Pneumonia). "
        "Answers provide **traceable bibliographic citations** with exact page and section numbers."
    )

    # Document Indexing & Management Sidebar/Expander
    with st.expander("📁 Indexed Guideline Document Inventory & Uploader", expanded=False):
        col_inv, col_up = st.columns([1, 1])
        with col_inv:
            st.markdown("##### Currently Indexed Guidelines")
            guidelines = list(settings.guidelines_dir.glob("*.*"))
            for g in guidelines:
                st.write(f"📄 **{g.stem.replace('_', ' ').title()}** (`{g.suffix}`)")
            st.caption(f"Total vector chunks currently in FAISS index: {len(retrieval_service.chunks)}")

        with col_up:
            st.markdown("##### Ingest New Guideline (.txt or .pdf)")
            uploaded_doc = st.file_uploader("Upload authorized guideline", type=["txt", "pdf", "md"])
            if uploaded_doc is not None:
                save_path = settings.guidelines_dir / uploaded_doc.name
                save_path.write_bytes(uploaded_doc.getvalue())
                st.success(f"Saved `{uploaded_doc.name}`. Click below to re-index.")
                if st.button("Re-Index Vector Database", type="secondary"):
                    with st.spinner("Re-chunking and building FAISS vector embeddings..."):
                        count = retrieval_service.reindex_all_guidelines()
                        st.success(f"Indexed {count} total chunks across all guidelines!")
                        st.rerun()

    # Search & Filter Controls
    filter_col1, filter_col2 = st.columns([2, 1])
    with filter_col1:
        doc_options = ["All Indexed Guidelines", "Chest Pain", "Heart Failure", "Pneumonia"]
        doc_filter_choice = st.selectbox("Filter Evidence Source:", doc_options)
        doc_filter_term = None if doc_filter_choice == "All Indexed Guidelines" else doc_filter_choice
    with filter_col2:
        top_k = st.slider("Max Evidence Chunks (Top-K):", min_value=1, max_value=6, value=3)

    # Query Input
    preset_q = st.selectbox("Quick Clinical Query Preset:", SAMPLE_QUESTIONS)
    default_text = "" if preset_q == SAMPLE_QUESTIONS[0] else preset_q
    
    query = st.text_input("Enter your clinical question:", value=default_text, placeholder="e.g. When is chest radiography indicated in acute chest pain evaluation?")

    ask_btn = st.button("Retrieve Evidence & Generate Answer →", type="primary", use_container_width=True)

    if ask_btn and query.strip():
        with st.spinner("Searching FAISS vector database and synthesizing grounded clinical evidence..."):
            answer: GuidelineAnswer = guideline_service.answer_query(
                query=query.strip(),
                document_filter=doc_filter_term,
                top_k=top_k
            )
            # Store in session state history
            if "guideline_history" not in st.session_state:
                st.session_state["guideline_history"] = []
            st.session_state["guideline_history"].append(answer)
            st.session_state["last_guideline_answer"] = answer
            st.toast("Guideline Evidence Retrieved!", icon="📚")

    # Display Current Answer
    if "last_guideline_answer" in st.session_state:
        ans: GuidelineAnswer = st.session_state["last_guideline_answer"]
        st.markdown("---")
        
        # Evidence sufficiency indicator
        if not ans.evidence_sufficient:
            st.error("⚠️ **INSUFFICIENT EVIDENCE DETECTED:**")
            st.warning(ans.answer)
            st.info("The application strictly prohibits inventing medical recommendations not present in the indexed guidelines.")
        else:
            st.subheader(f"Clinical Evidence Summary for: *'{ans.query}'*")
            st.markdown(
                f"<div style='background: #1e293b; padding: 20px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-bottom: 20px;'>"
                f"{ans.answer.replace(chr(10), '<br>')}"
                f"</div>", 
                unsafe_allow_html=True
            )

            # Citations Section
            st.subheader(f"Traceable Bibliographic Citations ({len(ans.citations)} Passages Retrieved)")
            for c in ans.citations:
                page_str = f"Page {c.page_number}" if c.page_number else "Page N/A"
                sec_str = f"Section: {c.section_heading}" if c.section_heading else "General"
                
                with st.expander(f"📌 [{c.citation_id}] {c.document_title} — {page_str} (Cosine Similarity: {c.similarity_score:.3f})"):
                    st.markdown(f"**Section:** `{sec_str}`")
                    st.markdown(f"**Exact Retrieved Passage:**")
                    st.markdown(f"> *\"{c.snippet}\"*")

            st.caption(f"🤖 Grounding Engine: `{ans.generator_model}` | 🔒 {ans.disclaimer}")
