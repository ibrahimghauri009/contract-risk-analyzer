"""Grounded Response Generation and Citation Synthesizer.

Synthesizes natural language answers strictly grounded in retrieved clauses,
extracting verbatim citations, character offsets, and risk assessments.
Works both with LLMs (OpenAI / local) and standalone deterministic synthesis.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from src.config import settings
from src.retriever import RetrievedClause
from src.classifier import ClauseClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GroundedAnswer:
    query: str
    answer: str
    risk_level: str
    risk_notes: str
    citations: List[Dict[str, Any]]
    retrieved_clauses: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class GroundedGenerator:
    """Generates grounded explanations and extracts exact verbatim citations."""

    def __init__(self, classifier: Optional[ClauseClassifier] = None):
        self.classifier = classifier or ClauseClassifier()

    def generate(self, query: str, retrieved_clauses: List[RetrievedClause]) -> GroundedAnswer:
        """
        Synthesizes a citation-grounded response from retrieved contract clauses.
        """
        if not retrieved_clauses:
            return GroundedAnswer(
                query=query,
                answer="No relevant clauses found in the contract for the specified query.",
                risk_level="LOW",
                risk_notes="No matching clauses were retrieved.",
                citations=[],
                retrieved_clauses=[]
            )

        citations = []
        top_clause = retrieved_clauses[0]

        # Extract structured citations from top retrieved chunks
        for idx, clause in enumerate(retrieved_clauses[:3]):
            citations.append({
                "citation_id": f"CIT-{idx+1:02d}",
                "chunk_id": clause.chunk_id,
                "quote": clause.text,
                "start_char": clause.start_char,
                "end_char": clause.end_char,
                "page_number": clause.page_number or 1,
                "section_title": clause.section_title or "Unlabeled Section"
            })

        # Run classifier on the top retrieved clause for risk context
        classification = self.classifier.predict(top_clause.text)
        category = classification["predicted_category"]

        # Deterministic Grounded Synthesis
        answer, risk_level, risk_notes = self._synthesize_grounded_summary(
            query=query,
            category=category,
            primary_clause=top_clause.text
        )

        return GroundedAnswer(
            query=query,
            answer=answer,
            risk_level=risk_level,
            risk_notes=risk_notes,
            citations=citations,
            retrieved_clauses=[c.to_dict() for c in retrieved_clauses]
        )

    def _synthesize_grounded_summary(
        self,
        query: str,
        category: str,
        primary_clause: str
    ) -> tuple[str, str, str]:
        """Synthesizes structured, legally grounded explanation based on clause type."""
        clause_clean = primary_clause.strip()
        
        if category == "Cap On Liability":
            return (
                f"The contract includes a Limitation of Liability clause capping aggregate damages. Cited terms: \"{clause_clean[:200]}...\"",
                "LOW",
                "Standard protective liability cap is in place."
            )
        elif category == "Uncapped Liability":
            return (
                f"The contract contains potential uncapped liability exposure without monetary limitation. Cited terms: \"{clause_clean[:200]}...\"",
                "CRITICAL",
                "High financial exposure risk: no aggregate liability ceiling detected."
            )
        elif category == "Non-Compete":
            return (
                f"A restrictive non-compete covenant is present restricting competitive business activities. Cited terms: \"{clause_clean[:200]}...\"",
                "HIGH",
                "Post-termination commercial restriction. Check geographical scope and duration."
            )
        elif category == "Termination For Convenience":
            return (
                f"The contract permits unilateral termination for convenience. Cited terms: \"{clause_clean[:200]}...\"",
                "MEDIUM",
                "Review advance notice requirements to prevent sudden contract termination."
            )
        elif category == "Governing Law":
            return (
                f"Governing jurisdiction and applicable law are specified. Cited terms: \"{clause_clean[:200]}...\"",
                "LOW",
                "Standard dispute resolution and governing law designated."
            )
        else:
            return (
                f"Relevant terms regarding '{query}' were identified in section '{category}'. Cited terms: \"{clause_clean[:200]}...\"",
                "MEDIUM" if "penalty" in clause_clean.lower() or "exclusive" in clause_clean.lower() else "LOW",
                f"Identified as '{category}' clause category."
            )
