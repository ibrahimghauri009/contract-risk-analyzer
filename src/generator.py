"""Grounded Response Generation and Citation Synthesizer.

Synthesizes natural language answers strictly grounded in retrieved clauses,
extracting verbatim citations, character offsets, and risk assessments.
Works with both LLMs (if API key is present) and smart extractive NLP synthesis.
"""
import re
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
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
                answer="No matching clauses were found in the uploaded contract for this query.",
                risk_level="LOW",
                risk_notes="No relevant text retrieved.",
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
                "section_title": clause.section_title or "Contract Clause"
            })

        # Run classifier on the top retrieved clause for category context
        classification = self.classifier.predict(top_clause.text)
        category = classification["predicted_category"]

        # Check if OpenAI API key is set for optional LLM generation
        api_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if api_key and len(api_key) > 10:
            answer, risk_level, risk_notes = self._generate_with_llm(query, retrieved_clauses, api_key)
        else:
            answer, risk_level, risk_notes = self._synthesize_extractive_answer(
                query=query,
                category=category,
                primary_clause=top_clause.text,
                all_clauses=retrieved_clauses
            )

        return GroundedAnswer(
            query=query,
            answer=answer,
            risk_level=risk_level,
            risk_notes=risk_notes,
            citations=citations,
            retrieved_clauses=[c.to_dict() for c in retrieved_clauses]
        )

    def _generate_with_llm(self, query: str, clauses: List[RetrievedClause], api_key: str) -> Tuple[str, str, str]:
        """Generates grounded answer via LLM API if key is provided."""
        try:
            import requests
            context_text = "\n\n".join([f"[Clause {i+1}]: {c.text}" for i, c in enumerate(clauses[:3])])
            prompt = f"""You are an expert legal contract analyst. Based ONLY on the contract excerpts below, answer the user's query with factual precision and cite exact terms.

Context:
{context_text}

Query: {query}

Provide a concise answer (2-3 sentences), identify the risk level (LOW, MEDIUM, HIGH, or CRITICAL), and give a short 1-sentence risk note.
Format:
ANSWER: <your grounded answer>
RISK_LEVEL: <LOW/MEDIUM/HIGH/CRITICAL>
RISK_NOTES: <short risk note>"""

            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": settings.LLM_MODEL or "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"]
                # Parse
                ans = ""
                r_level = "LOW"
                r_notes = ""
                for line in content.split("\n"):
                    if line.startswith("ANSWER:"):
                        ans = line.replace("ANSWER:", "").strip()
                    elif line.startswith("RISK_LEVEL:"):
                        r_level = line.replace("RISK_LEVEL:", "").strip().upper()
                    elif line.startswith("RISK_NOTES:"):
                        r_notes = line.replace("RISK_NOTES:", "").strip()
                if ans:
                    return ans, r_level, r_notes
        except Exception as e:
            logger.warning(f"LLM generation fallback: {e}")

        return self._synthesize_extractive_answer(query, "General", clauses[0].text, clauses)

    def _synthesize_extractive_answer(
        self,
        query: str,
        category: str,
        primary_clause: str,
        all_clauses: List[RetrievedClause]
    ) -> Tuple[str, str, str]:
        """
        Extractive, dynamic answer synthesis that directly extracts key facts,
        numbers, durations, and terms from the retrieved contract text.
        """
        text_clean = primary_clause.strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.\?!;])\s+", text_clean) if s.strip()]
        
        # Find the sentence in the clause with highest keyword overlap with user query
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        best_sentence = sentences[0] if sentences else text_clean
        best_score = -1

        for s in sentences:
            s_words = set(re.findall(r"\b\w{3,}\b", s.lower()))
            overlap = len(query_words.intersection(s_words))
            if overlap > best_score:
                best_score = overlap
                best_sentence = s

        # Extract specific patterns if present
        durations = re.findall(r"\b(?:\d+|one|two|three|four|five|six|twelve|24|30|60|90)\s*(?:years?|months?|days?|weeks?|hours?)\b", text_clean, re.I)
        amounts = re.findall(r"\$[\d,]+(?:\.\d+)?|\b\d+\s*(?:percent|%|USD|dollars)\b", text_clean, re.I)
        jurisdictions = re.findall(r"\b(?:State of \w+|laws of \w+|Delaware|New York|California|England|Texas)\b", text_clean, re.I)

        details = []
        if durations:
            details.append(f"Specified period: **{', '.join(set(durations))}**")
        if amounts:
            details.append(f"Financial terms: **{', '.join(set(amounts))}**")
        if jurisdictions:
            details.append(f"Jurisdiction: **{', '.join(set(jurisdictions))}**")

        detail_str = f" ({'; '.join(details)})" if details else ""

        # Determine risk
        text_lower = text_clean.lower()
        if "unlimited liability" in text_lower or "uncapped" in text_lower or ("no" in text_lower and "limitation of liability" in text_lower):
            risk_level = "CRITICAL"
            risk_notes = "High financial exposure: potential uncapped liability detected."
        elif "non-compete" in text_lower or "liquidated damages" in text_lower or "exclusive" in text_lower:
            risk_level = "HIGH"
            risk_notes = f"Restrictive commercial constraint ({category})."
        elif "terminate" in text_lower or "indemnif" in text_lower:
            risk_level = "MEDIUM"
            risk_notes = f"Operational term under {category}."
        else:
            risk_level = "LOW"
            risk_notes = f"Standard clause identified as {category}."

        answer = f"Based on the contract, {best_sentence}{detail_str}"

        return answer, risk_level, risk_notes
