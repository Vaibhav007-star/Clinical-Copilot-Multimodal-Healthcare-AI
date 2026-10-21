"""
Clinical document chunker and text extraction module.
Preserves page numbers, clinical section headings, and bibliographic metadata.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from modules.rag.schemas import DocumentChunk

def extract_text_from_file(file_path: Path) -> List[Tuple[int, str]]:
    """
    Extracts text per page from PDF or text file.
    Returns list of (page_number, text).
    """
    pages: List[Tuple[int, str]] = []
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf":
        if not HAS_PYPDF:
            return [(1, f"[PDF Document {file_path.name}]")]
        reader = PdfReader(str(file_path))
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((page_idx + 1, text.strip()))
    else:
        # Plain text or markdown
        content = file_path.read_text(encoding="utf-8", errors="replace")
        # Check for [Page X] markers if present
        page_splits = re.split(r'\[Page\s+(\d+)\]', content)
        if len(page_splits) > 1:
            # We have explicit page markers
            # page_splits[0] is preamble
            for i in range(1, len(page_splits), 2):
                p_num = int(page_splits[i])
                p_text = page_splits[i+1].strip()
                pages.append((p_num, p_text))
        else:
            pages.append((1, content.strip()))
            
    return pages

def chunk_clinical_document(
    file_path: Path,
    chunk_size: int = 400,
    chunk_overlap: int = 60
) -> List[DocumentChunk]:
    """
    Splits clinical document into semantic chunks while retaining page and section context.
    """
    doc_title = file_path.stem.replace("_", " ").title()
    pages = extract_text_from_file(file_path)
    chunks: List[DocumentChunk] = []
    chunk_idx = 0

    for page_num, page_text in pages:
        lines = page_text.splitlines()
        current_section = "General Clinical Guidance"
        current_chunk_words: List[str] = []
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
                
            # Check if this line is a heading
            if line_str.startswith("#") or re.match(r'^\d+\.\d+\s+', line_str):
                current_section = line_str.lstrip("#").strip()
                continue
                
            words = line_str.split()
            current_chunk_words.extend(words)
            
            if len(current_chunk_words) >= chunk_size:
                chunk_text = " ".join(current_chunk_words)
                chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_idx}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    document_title=doc_title,
                    page_number=page_num,
                    section_heading=current_section,
                    text=chunk_text,
                    token_count=len(current_chunk_words)
                ))
                chunk_idx += 1
                current_chunk_words = current_chunk_words[-chunk_overlap:]
                
        # Remainder words on this page
        if current_chunk_words:
            chunk_text = " ".join(current_chunk_words)
            chunk_id = f"{file_path.stem}_p{page_num}_c{chunk_idx}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                document_title=doc_title,
                page_number=page_num,
                section_heading=current_section,
                text=chunk_text,
                token_count=len(current_chunk_words)
            ))
            chunk_idx += 1

    return chunks
