"""Legal Risk Rules and Scoring Engine.

Combines:
1. Supervised Classifier predictions
2. Deep semantic pattern analysis & heuristic risk extractors
3. Missing protective clause checks
4. Exact verbatim citation evidence mapping
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
        across all chunks of the contract.
        """
        findings: List[RiskFinding] = []
        detected_categories = set()
        
        if not chunks:
            return ContractRiskReport(
                contract_id=contract_id,
                overall_score=0,
                risk_level=RiskSeverity.LOW,
                total_clauses_analyzed=0,
                findings=[],
                detected_categories=[],
                missing_protective_clauses=[]
            )

        # 1. Classify each chunk and collect semantic observations
        classified_chunks = []
        for chunk in chunks:
            pred = self.classifier.predict(chunk.text)
            cat = pred["predicted_category"]
            if pred["is_confident"] and cat != "Uncategorized/General":
                detected_categories.add(cat)
                
            classified_chunks.append({
                "chunk": chunk,
                "category": cat,
                "confidence": pred["confidence"]
            })

        # 2. Deep Rule-Based & Semantic Risk Scanners across all chunks
        for item in classified_chunks:
            chunk = item["chunk"]
            cat = item["category"]
            text_raw = chunk.text
            text_lower = text_raw.lower()

            # Rule A: Uncapped / Unlimited Liability
            if cat == "Uncapped Liability" or ("unlimited liability" in text_lower) or ("no limitation of liability" in text_lower) or ("neither party shall be subject to any monetary limitation" in text_lower):
                findings.append(RiskFinding(
                    category="Liability",
                    severity=RiskSeverity.CRITICAL,
                    title="Uncapped / Unlimited Liability Exposure",
                    description="The contract accepts unlimited or uncapped financial liability for direct or consequential losses without a monetary ceiling.",
                    recommendation="Insert a mutual aggregate liability cap (e.g., total fees paid in preceding 12 months).",
                    citation_text=text_raw,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

            # Rule B: Non-Compete Restrictions
            elif cat == "Non-Compete" or ("non-compete" in text_lower) or ("shall not engage" in text_lower and "compet" in text_lower):
                long_duration = bool(re.search(r"(?:two|three|four|five|2|3|4|5)\s*(?:years?|yr)", text_lower))
                worldwide = "worldwide" in text_lower or "anywhere in the world" in text_lower
                
                desc = "Restricts future commercial or employment activities in competing industries."
                if long_duration:
                    desc += " Detected extended duration (>1 year)."
                if worldwide:
                    desc += " Detected broad worldwide geographical restriction."

                findings.append(RiskFinding(
                    category="Restrictive Covenants",
                    severity=RiskSeverity.HIGH if (long_duration or worldwide) else RiskSeverity.MEDIUM,
                    title=f"Non-Compete Covenant {'(High Severity)' if (long_duration or worldwide) else ''}",
                    description=desc,
                    recommendation="Narrow the geographic scope and limit non-compete duration to 6-12 months maximum.",
                    citation_text=text_raw,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

            # Rule C: Exclusivity Obligations
            elif cat == "Exclusivity" or ("exclusive partner" in text_lower) or ("strictly prohibited from offering" in text_lower) or ("exclusive commercial" in text_lower):
                findings.append(RiskFinding(
                    category="Commercial Constraints",
                    severity=RiskSeverity.HIGH,
                    title="Exclusivity Lock-In Clause",
                    description="Mandates exclusive commercial dealing, preventing transactions or partnerships with other vendors or competitors.",
                    recommendation="Review commercial impact; add minimum commitment volume thresholds or carve-outs.",
                    citation_text=text_raw,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

            # Rule D: Liquidated Damages / Penalties
            elif cat == "Liquidated Damages" or ("liquidated damages" in text_lower) or ("penalty of" in text_lower) or ("per day of delay" in text_lower):
                findings.append(RiskFinding(
                    category="Financial Penalties",
                    severity=RiskSeverity.HIGH,
                    title="Liquidated Damages / Daily Penalties",
                    description="Predetermined financial penalties apply automatically upon delay or breach without requiring proof of actual loss.",
                    recommendation="Ensure damages reflect genuine pre-estimated losses and add a reasonable grace/cure period.",
                    citation_text=text_raw,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

            # Rule E: Unilateral / Immediate Termination
            elif cat == "Termination For Convenience" or ("terminate" in text_lower and "convenience" in text_lower):
                short_notice = bool(re.search(r"(?:immediate|without prior notice|under\s*10\s*days|7\s*days|48\s*hours)", text_lower))
                if short_notice:
                    findings.append(RiskFinding(
                        category="Termination Exposure",
                        severity=RiskSeverity.HIGH,
                        title="Immediate Termination for Convenience",
                        description="Allows one party to cancel the contract immediately without cause or notice.",
                        recommendation="Negotiate a minimum 30-to-60 days prior written notice requirement.",
                        citation_text=text_raw,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        chunk_id=chunk.chunk_id
                    ))

            # Rule F: Broad Unilateral Indemnification
            elif "indemnif" in text_lower and ("hold harmless" in text_lower or "defend" in text_lower):
                if "customer shall indemnify" in text_lower or "provider accepts" in text_lower or "unlimited" in text_lower:
                    findings.append(RiskFinding(
                        category="Indemnity Obligations",
                        severity=RiskSeverity.MEDIUM,
                        title="Broad Indemnification Obligation",
                        description="Requires indemnifying and defending the other party against third-party claims.",
                        recommendation="Ensure indemnity is mutual, capped, and excludes gross negligence or willful misconduct.",
                        citation_text=text_raw,
                        start_char=chunk.start_char,
                        end_char=chunk.end_char,
                        chunk_id=chunk.chunk_id
                    ))

            # Rule G: Automatic Renewal / Evergreen Lock-in
            elif "automatic renewal" in text_lower or "automatically renew" in text_lower or "evergreen" in text_lower:
                findings.append(RiskFinding(
                    category="Term & Renewal",
                    severity=RiskSeverity.LOW,
                    title="Automatic Contract Renewal",
                    description="Agreement automatically extends unless active notice of non-renewal is provided before the deadline.",
                    recommendation="Calendar the non-renewal notice deadline (typically 30-60 days before expiration).",
                    citation_text=text_raw,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_id=chunk.chunk_id
                ))

        # 3. Missing Protective Clauses Rules (only evaluated on comprehensive contracts > 3 chunks)
        missing_protections = []
        full_contract_text = " ".join([c.text.lower() for c in chunks])
        
        if len(chunks) >= 3:
            # Check: Missing Liability Cap
            has_cap = "Cap On Liability" in detected_categories or "limitation of liability" in full_contract_text or "aggregate liability" in full_contract_text
            if not has_cap:
                missing_protections.append("Cap On Liability")
                findings.append(RiskFinding(
                    category="Missing Protection",
                    severity=RiskSeverity.HIGH,
                    title="Missing Limitation of Liability Clause",
                    description="No explicit liability cap was identified, leaving parties potentially exposed to broad statutory damages.",
                    recommendation="Add a standard limitation of liability section with aggregate fee-based cap."
                ))

            # Check: Missing Governing Law
            has_law = "Governing Law" in detected_categories or "governing law" in full_contract_text or "jurisdiction" in full_contract_text or "laws of the state" in full_contract_text
            if not has_law:
                missing_protections.append("Governing Law")
                findings.append(RiskFinding(
                    category="Missing Protection",
                    severity=RiskSeverity.MEDIUM,
                    title="Missing Governing Law & Jurisdiction",
                    description="No standard governing law or dispute venue clause detected.",
                    recommendation="Specify applicable state/national jurisdiction to prevent cross-border venue disputes."
                ))

            # Check: Missing Anti-Assignment
            has_assignment = "Anti-Assignment" in detected_categories or "assignment" in full_contract_text or "assign this agreement" in full_contract_text
            if not has_assignment:
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
            RiskSeverity.HIGH: 25,
            RiskSeverity.CRITICAL: 40
        }
        raw_score = sum(score_weights.get(f.severity, 10) for f in findings)
        overall_score = min(100, max(0, raw_score))

        if overall_score >= 60 or any(f.severity == RiskSeverity.CRITICAL for f in findings):
            overall_level = RiskSeverity.CRITICAL if overall_score >= 75 else RiskSeverity.HIGH
        elif overall_score >= 25:
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
