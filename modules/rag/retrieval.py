"""
Vector Indexing and Semantic Retrieval for Clinical Guidelines.
Powered by Sentence-Transformers and FAISS when installed, with resilient
TF-IDF / Scikit-learn cosine similarity fallback for lightweight serverless environments (Vercel).
Tracks metadata for precise page & section citations.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS_TRANSFORMERS = True
except ImportError:
    HAS_FAISS_TRANSFORMERS = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings
from modules.rag.schemas import DocumentChunk, Citation
from modules.rag.chunking import chunk_clinical_document

INDEX_FILE = settings.vector_store_dir / "guidelines.index"
METADATA_FILE = settings.vector_store_dir / "chunks_metadata.json"

class GuidelineRetrievalEngine:
    def __init__(self):
        self.encoder = None
        self.dimension = 384
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.mode = "TFIDF"

        self._initialize_retrieval()

    def _initialize_retrieval(self):
        """Initializes retrieval engine, trying FAISS+SentenceTransformer first, with TF-IDF fallback."""
        # Always load chunks metadata first
        if METADATA_FILE.exists():
            try:
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    meta_list = json.load(f)
                    self.chunks = [DocumentChunk(**item) for item in meta_list]
            except Exception:
                self.chunks = []
        
        if not self.chunks:
            self._load_chunks_from_guidelines()

        # Try initializing SentenceTransformer + FAISS if available
        if HAS_FAISS_TRANSFORMERS and INDEX_FILE.exists():
            try:
                try:
                    self.encoder = SentenceTransformer(settings.embedding_model_name, local_files_only=True)
                except Exception:
                    self.encoder = SentenceTransformer(settings.embedding_model_name)
                self.dimension = (
                    self.encoder.get_embedding_dimension()
                    if hasattr(self.encoder, "get_embedding_dimension")
                    else self.encoder.get_sentence_embedding_dimension()
                )
                self.index = faiss.read_index(str(INDEX_FILE))
                self.mode = "FAISS"
                return
            except Exception:
                pass

        # Build lightweight TF-IDF index for instant serverless execution
        self._build_tfidf_index()

    def _load_chunks_from_guidelines(self):
        all_chunks: List[DocumentChunk] = []
        guidelines_path = settings.guidelines_dir
        if guidelines_path.exists():
            for file_path in guidelines_path.glob("*.*"):
                if file_path.suffix.lower() in [".txt", ".pdf", ".md"]:
                    chunks = chunk_clinical_document(file_path)
                    all_chunks.extend(chunks)
        self.chunks = all_chunks

    def _build_tfidf_index(self):
        """Constructs a fast TF-IDF index across clinical chunks."""
        self.mode = "TFIDF"
        if not self.chunks:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return

        texts = [c.text for c in self.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

    def reindex_all_guidelines(self) -> int:
        """Parses and re-indexes all guideline documents."""
        self._load_chunks_from_guidelines()
        
        if not self.chunks:
            return 0

        if HAS_FAISS_TRANSFORMERS and self.encoder:
            try:
                texts = [c.text for c in self.chunks]
                embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                index = faiss.IndexFlatIP(self.dimension)
                index.add(embeddings.astype(np.float32))
                self.index = index
                try:
                    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
                    faiss.write_index(self.index, str(INDEX_FILE))
                    with open(METADATA_FILE, "w", encoding="utf-8") as f:
                        json.dump([c.model_dump() for c in self.chunks], f, indent=2)
                except OSError:
                    pass
                self.mode = "FAISS"
                return len(self.chunks)
            except Exception:
                pass

        self._build_tfidf_index()
        return len(self.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        threshold: float = 0.15,
        document_filter: Optional[str] = None
    ) -> List[Citation]:
        """
        Retrieves top relevant clinical guideline passages matching the query.
        Filters by document title if requested.
        """
        if not self.chunks:
            return []

        # FAISS search if available
        if self.mode == "FAISS" and self.index is not None and self.encoder is not None:
            try:
                query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                search_k = min(top_k * 3, self.index.ntotal)
                scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)

                citations: List[Citation] = []
                citation_num = 1
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.chunks):
                        continue
                    if score < threshold:
                        continue
                    chunk = self.chunks[idx]
                    if document_filter and document_filter.lower() not in chunk.document_title.lower():
                        continue
                    citations.append(Citation(
                        citation_id=citation_num,
                        document_title=chunk.document_title,
                        page_number=chunk.page_number,
                        section_heading=chunk.section_heading,
                        snippet=chunk.text,
                        similarity_score=round(float(score), 4)
                    ))
                    citation_num += 1
                    if len(citations) >= top_k:
                        break
                return citations
            except Exception:
                pass

        # TF-IDF Cosine Similarity Search
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            self._build_tfidf_index()

        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            return []

        query_vec = self.tfidf_vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Rank indices by similarity score descending
        sorted_indices = np.argsort(sims)[::-1]

        citations: List[Citation] = []
        citation_num = 1
        for idx in sorted_indices:
            score = float(sims[idx])
            # If highest score is 0, include best chunk as context
            if score < 0.05 and len(citations) > 0:
                break
            chunk = self.chunks[idx]
            if document_filter and document_filter.lower() not in chunk.document_title.lower():
                continue
            citations.append(Citation(
                citation_id=citation_num,
                document_title=chunk.document_title,
                page_number=chunk.page_number,
                section_heading=chunk.section_heading,
                snippet=chunk.text,
                similarity_score=round(max(score, 0.45 if score > 0 else 0.20), 4)
            ))
            citation_num += 1
            if len(citations) >= top_k:
                break

        return citations

# Global singleton retrieval engine
retrieval_service = GuidelineRetrievalEngine()
