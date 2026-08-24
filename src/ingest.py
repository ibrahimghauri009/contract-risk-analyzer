"""Legal Contract Ingestion and Structure-Aware Chunking.

Handles:
- PDF and Text document ingestion with line normalization
- Section and clause-aware segmentation
- Exact character offset tracking for verifiable citations
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pypdf import PdfReader

@dataclass
class ContractChunk:
    chunk_id: str
    contract_id: str
    text: str
    start_char: int
    end_char: int
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ContractIngestion:
    """Ingests raw contract documents (PDF or TXT) and segments them into coherent clause units."""

    # Regex patterns for legal section headers (e.g., "Section 1.2", "ARTICLE IV", "1. Definitions", "1.1 Termination")
    SECTION_PATTERN = re.compile(
        r"(?i)^(?:section|article|clause)?\s*(\d+(?:\.\d+)*|[IVXLCDM]+)[\.\:\s\-]+([A-Z][^\n]+)?",
        re.MULTILINE
    )

    def load_document(self, file_path: str | Path) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Loads document content and per-page metadata.
        Returns full text and page mapping metadata.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Contract file not found: {path}")

        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        else:
            return self._load_text(path)

    def _load_text(self, path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        page_maps = [{"page": 1, "start_char": 0, "end_char": len(text)}]
        return text, page_maps

    def _load_pdf(self, path: Path) -> Tuple[str, List[Dict[str, Any]]]:
        reader = PdfReader(path)
        full_text_parts = []
        page_maps = []
        current_offset = 0

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            # Clean weird PDF line breaks while keeping paragraph breaks
            cleaned_page = self._clean_pdf_text(page_text)
            start = current_offset
            end = start + len(cleaned_page)
            full_text_parts.append(cleaned_page)
            page_maps.append({
                "page": idx + 1,
                "start_char": start,
                "end_char": end
            })
            current_offset = end + 2  # double newline separator

        return "\n\n".join(full_text_parts), page_maps

    def _clean_pdf_text(self, text: str) -> str:
        """Fixes hyphenated line-breaks and joins broken lines inside paragraphs."""
        # Fix hyphenated words broken across lines (e.g. "agree-\nment" -> "agreement")
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Normalize multiple newlines
        text = re.sub(r"\r\n|\r", "\n", text)
        # If line ends without period/colon and next line starts with lowercase, join them
        text = re.sub(r"(?<![\.\:\;\?\!])\n(?=[a-z])", " ", text)
        return text

    def chunk_contract(
        self,
        text: str,
        contract_id: str,
        page_maps: Optional[List[Dict[str, Any]]] = None,
        max_chunk_chars: int = 800,
        min_chunk_chars: int = 60
    ) -> List[ContractChunk]:
        """
        Chunks text respecting legal paragraphs and section boundaries
        while maintaining exact start and end offsets.
        """
        if not text.strip():
            return []

        # Split on paragraph breaks or section headers
        # Split by double newlines or single newlines followed by numbers/sections
        raw_sections = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        
        # If text doesn't have double newlines, split by line or section
        if len(raw_sections) <= 2 and len(text) > 600:
            raw_sections = [s.strip() for s in re.split(r"(?=\n\s*(?:(?:Section|Article|Clause|\d+\.|\d+\.\d+)\s+))", text) if s.strip()]
            if len(raw_sections) <= 1:
                # Fallback to splitting by sentence groups
                raw_sections = [s.strip() for s in re.split(r"(?<=\.)\s+(?=[A-Z0-9])", text) if s.strip()]

        chunks: List[ContractChunk] = []
        current_offset = 0
        chunk_idx = 0
        
        current_section_num = None
        current_section_title = None

        for section_text in raw_sections:
            if not section_text:
                continue

            # Find actual start index of this section in original text
            start_pos = text.find(section_text, current_offset)
            if start_pos == -1:
                start_pos = current_offset
            end_pos = start_pos + len(section_text)
            current_offset = end_pos

            # Check if this section starts with a Section header
            header_match = self.SECTION_PATTERN.match(section_text)
            if header_match:
                current_section_num = header_match.group(1)
                current_section_title = header_match.group(2).strip() if header_match.group(2) else None

            # Determine page number
            page_num = 1
            if page_maps:
                for pm in page_maps:
                    if pm["start_char"] <= start_pos <= pm["end_char"]:
                        page_num = pm["page"]
                        break

            # If section is very long, split into sub-chunks on sentence boundaries
            if len(section_text) > max_chunk_chars:
                sub_chunks = self._split_long_paragraph(
                    section_text, start_pos, contract_id, chunk_idx,
                    current_section_num, current_section_title, page_num, max_chunk_chars
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
            elif len(section_text) >= min_chunk_chars:
                chunk = ContractChunk(
                    chunk_id=f"{contract_id}_chunk_{chunk_idx:04d}",
                    contract_id=contract_id,
                    text=section_text,
                    start_char=start_pos,
                    end_char=end_pos,
                    section_number=current_section_num,
                    section_title=current_section_title,
                    page_number=page_num
                )
                chunks.append(chunk)
                chunk_idx += 1

        # Fallback if no chunks met min_chunk_chars
        if not chunks and text.strip():
            chunks.append(ContractChunk(
                chunk_id=f"{contract_id}_chunk_0000",
                contract_id=contract_id,
                text=text.strip(),
                start_char=0,
                end_char=len(text.strip()),
                page_number=1
            ))

        return chunks

    def _split_long_paragraph(
        self,
        para_text: str,
        base_offset: int,
        contract_id: str,
        start_chunk_idx: int,
        section_num: Optional[str],
        section_title: Optional[str],
        page_num: int,
        max_chars: int
    ) -> List[ContractChunk]:
        """Splits an oversized paragraph on sentence boundaries while preserving offsets."""
        sentences = [s.strip() for s in re.split(r"(?<=[.\?!;])\s+", para_text) if s.strip()]
        sub_chunks = []
        curr_text = ""
        curr_start = base_offset
        c_idx = start_chunk_idx

        for sent in sentences:
            if len(curr_text) + len(sent) > max_chars and curr_text:
                curr_end = curr_start + len(curr_text.strip())
                sub_chunks.append(ContractChunk(
                    chunk_id=f"{contract_id}_chunk_{c_idx:04d}",
                    contract_id=contract_id,
                    text=curr_text.strip(),
                    start_char=curr_start,
                    end_char=curr_end,
                    section_number=section_num,
                    section_title=section_title,
                    page_number=page_num
                ))
                c_idx += 1
                curr_start = curr_end + 1
                curr_text = sent + " "
            else:
                curr_text += sent + " "

        if curr_text.strip():
            curr_end = curr_start + len(curr_text.strip())
            sub_chunks.append(ContractChunk(
                chunk_id=f"{contract_id}_chunk_{c_idx:04d}",
                contract_id=contract_id,
                text=curr_text.strip(),
                start_char=curr_start,
                end_char=curr_end,
                section_number=section_num,
                section_title=section_title,
                page_number=page_num
            ))

        return sub_chunks
