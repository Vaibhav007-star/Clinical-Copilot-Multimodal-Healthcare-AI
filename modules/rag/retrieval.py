"""
Vector Indexing and Semantic Retrieval for Clinical Guidelines.
Powered by Sentence-Transformers and FAISS, with metadata tracking for page & section citations.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config.settings import settings
from modules.rag.schemas import DocumentChunk, Citation
from modules.rag.chunking import chunk_clinical_document

INDEX_FILE = settings.vector_store_dir / "guidelines.index"
METADATA_FILE = settings.vector_store_dir / "chunks_metadata.json"

class GuidelineRetrievalEngine:
    def __init__(self):
        self.encoder = SentenceTransformer(settings.embedding_model_name)
        self.dimension = (
            self.encoder.get_embedding_dimension()
            if hasattr(self.encoder, "get_embedding_dimension")
            else self.encoder.get_sentence_embedding_dimension()
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[DocumentChunk] = []
        self._load_or_build_index()

    def _load_or_build_index(self):
        """Loads pre-built FAISS index or builds from documents in data/guidelines."""
        if INDEX_FILE.exists() and METADATA_FILE.exists():
            try:
                self.index = faiss.read_index(str(INDEX_FILE))
                with open(METADATA_FILE, "r", encoding="utf-8") as f:
                    meta_list = json.load(f)
                    self.chunks = [DocumentChunk(**item) for item in meta_list]
                return
            except Exception:
                pass
        
        # Build index from guidelines directory
        self.reindex_all_guidelines()

    def reindex_all_guidelines(self) -> int:
        """Parses and indexes all guideline documents in the guidelines directory."""
        all_chunks: List[DocumentChunk] = []
        guidelines_path = settings.guidelines_dir
        
        for file_path in guidelines_path.glob("*.*"):
            if file_path.suffix.lower() in [".txt", ".pdf", ".md"]:
                chunks = chunk_clinical_document(file_path)
                all_chunks.extend(chunks)

        if not all_chunks:
            # Empty fallback index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            return 0

        # Encode chunks
        texts = [c.text for c in all_chunks]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Build FAISS inner product index
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings.astype(np.float32))

        self.index = index
        self.chunks = all_chunks

        # Persist index and metadata
        settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in self.chunks], f, indent=2)

        return len(all_chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        threshold: float = 0.25,
        document_filter: Optional[str] = None
    ) -> List[Citation]:
        """
        Retrieves top relevant clinical guideline passages matching the query.
        Filters by document title if requested.
        """
        if self.index is None or len(self.chunks) == 0:
            self._load_or_build_index()
            
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search for more candidates to account for document filtering
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

# Global singleton retrieval engine
retrieval_service = GuidelineRetrievalEngine()
