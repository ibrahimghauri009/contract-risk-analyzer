"""Legal Risk Rules and Scoring Engine.

Combines:
1. Supervised Classifier predictions
2. Deterministic legal risk heuristics & missing-clause checks
3. Structured Risk Assessment (0-100 score + risk level + itemized findings with exact citations)
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from src.ingest import ContractChunk
from src.classifier import ClauseClassifier
from src.retriever import RetrievedClause

class RiskSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskFinding:
    category: str
    severity: RiskSeverity
    title: str
    description: str
    recommendation: str
    citation_text: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    chunk_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d

@dataclass
class ContractRiskReport:
    contract_id: str
    overall_score: int  # 0 to 100
    risk_level: RiskSeverity
    total_clauses_analyzed: int
    findings: List[RiskFinding]
    detected_categories: List[str]
    missing_protective_clauses: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level.value,
            "total_clauses_analyzed": self.total_clauses_analyzed,
            "findings": [f.to_dict() for f in self.findings],
            "detected_categories": self.detected_categories,
            "missing_protective_clauses": self.missing_protective_clauses
        }

class RiskEngine:
    """Evaluates contract risk using classifier findings and legal domain rules."""

    def __init__(self, classifier: Optional[ClauseClassifier] = None):
        self.classifier = classifier or ClauseClassifier()

    def analyze_contract(self, chunks: List[ContractChunk], contract_id: str = "contract") -> ContractRiskReport:
        """
        Runs comprehensive clause classification and rule-based risk evaluation
        across all chunks of a contract.
        """
        findings: List[RiskFinding] = []
        detected_categories = set()
        
        # 1. Classify each chunk
        classified_chunks = []
        for chunk in chunks:
            pred = self.classifier.predict(chunk.text)
            if pred["is_confident"]:
                cat = pred["predicted_category"]
                detected_categories.add(cat)
                classified_chunks.append({
                    "chunk": chunk,
                    "category": cat,
                    "confidence": pred["confidence"]
                })

        # 2. Evaluate Specific Clause Risks
        for item in classified_chunks:
            chunk = item["chunk"]
            cat = item["category"]
            text_lower = chunk.text.lower()

            # Rule A: Uncapped Liability
            if cat == "Uncapped Liability" or ("unlimited liability" in text_lower or "no limitation of liability" in text_lower):
                findings.append(RiskFinding(
                    category="Liability",
                    severity=RiskSeverity.CRITICAL,
                    title="Uncapped / Unlimited Liability Exposure",
                    description="The contract may expose one or both parties to uncapped damages with no monetary ceiling.",
                    recommendation="Insert a mutual aggregate liability cap (e.g., total fees paid in preceding 12 months).",
                    citation_text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

            # Rule B: Non-Compete Restrictions
            elif cat == "Non-Compete":
                # Check for duration > 1 year
                long_duration = bool(re.search(r"(?:two|three|four|five|2|3|4|5)\s*(?:years|yr)", text_lower))
                findings.append(RiskFinding(
                    category="Restrictive Covenants",
                    severity=RiskSeverity.HIGH if long_duration else RiskSeverity.MEDIUM,
                    title=f"Non-Compete Restriction {'(Extended Duration)' if long_duration else ''}",
                    description="Restricts future business activities and competitive engagements after termination.",
                    recommendation="Narrow the geographic scope and limit duration to 6-12 months maximum.",
                    citation_text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

            # Rule C: Exclusivity Obligations
            elif cat == "Exclusivity":
                findings.append(RiskFinding(
                    category="Commercial Constraints",
                    severity=RiskSeverity.HIGH,
                    title="Exclusivity Clause Detected",
                    description="Mandates exclusive commercial dealing, preventing partnerships with other vendors or clients.",
                    recommendation="Review commercial impact; add minimum commitment requirements or carve-outs.",
                    citation_text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

            # Rule D: Liquidated Damages / Heavy Penalties
            elif cat == "Liquidated Damages":
                findings.append(RiskFinding(
                    category="Financial Penalties",
                    severity=RiskSeverity.HIGH,
                    title="Liquidated Damages / Fixed Penalties",
                    description="Predetermined financial penalties apply in the event of default or delay.",
                    recommendation="Ensure damages reflect genuine pre-estimated losses rather than punitive fines.",
                    citation_text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

            # Rule E: Termination for Convenience
            elif cat == "Termination For Convenience":
                short_notice = bool(re.search(r"(?:immediate|immediate notice|under\s*10\s*days|7\s*days|5\s*days)", text_lower))
                if short_notice:
                    findings.append(RiskFinding(
                        category="Termination Exposure",
                        severity=RiskSeverity.MEDIUM,
                        title="Short-Notice Termination for Convenience",
                        description="Allows early unilateral cancellation on very short notice without cause.",
                        recommendation="Negotiate a minimum 30-to-60 days prior written notice requirement.",
                        citation_text=chunk.text,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        chunk_id=chunk.chunk_id
                    ))

        # 3. Missing Protective Clauses Rules
        missing_protections = []
        
        # Check: Missing Liability Cap
        if "Cap On Liability" not in detected_categories and "Uncapped Liability" not in detected_categories:
            missing_protections.append("Cap On Liability")
            findings.append(RiskFinding(
                category="Missing Protection",
                severity=RiskSeverity.HIGH,
                title="Missing Limitation of Liability Clause",
                description="No explicit liability cap was identified, leaving parties potentially exposed to broad damages.",
                recommendation="Add a standard limitation of liability section with aggregate dollar or fee-based cap."
            ))

        # Check: Missing Governing Law
        if "Governing Law" not in detected_categories:
            missing_protections.append("Governing Law")
            findings.append(RiskFinding(
                category="Missing Protection",
                severity=RiskSeverity.MEDIUM,
                title="Missing Governing Law & Jurisdiction",
                description="No standard governing law or dispute venue clause detected.",
                recommendation="Specify applicable state/national jurisdiction to prevent cross-border venue disputes."
            ))

        # Check: Missing Anti-Assignment
        if "Anti-Assignment" not in detected_categories:
            missing_protections.append("Anti-Assignment")
            findings.append(RiskFinding(
                category="Missing Protection",
                severity=RiskSeverity.LOW,
                title="Missing Anti-Assignment Clause",
                description="Contract lacks restrictions against unauthorized assignment of obligations to third parties.",
                recommendation="Include requirement for written consent before transferring rights."
            ))

        # 4. Compute Overall Risk Score (0-100)
        score_weights = {
            RiskSeverity.LOW: 5,
            RiskSeverity.MEDIUM: 15,
            RiskSeverity.HIGH: 30,
            RiskSeverity.CRITICAL: 50
        }
        raw_score = sum(score_weights.get(f.severity, 10) for f in findings)
        overall_score = min(100, raw_score)

        if overall_score >= 60 or any(f.severity == RiskSeverity.CRITICAL for f in findings):
            overall_level = RiskSeverity.CRITICAL if overall_score >= 80 else RiskSeverity.HIGH
        elif overall_score >= 30:
            overall_level = RiskSeverity.MEDIUM
        else:
            overall_level = RiskSeverity.LOW

        return ContractRiskReport(
            contract_id=contract_id,
            overall_score=overall_score,
            risk_level=overall_level,
            total_clauses_analyzed=len(chunks),
            findings=findings,
            detected_categories=sorted(list(detected_categories)),
            missing_protective_clauses=missing_protections
        )
