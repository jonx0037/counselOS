"""
Seed the knowledge base with mock corporate legal reference documents.
Run once before starting the backend:

    python -m rag.seed_data
"""
import logging
from rag.store import add_documents, document_count

logger = logging.getLogger(__name__)

DOCUMENTS = [
    {"id": "contract-001", "text": "Contract Review Policy: All commercial contracts exceeding $500,000 in value require partner-level review. NDAs and standard service agreements under $50,000 may be handled by senior associates. Force majeure clauses, indemnification terms, and limitation-of-liability provisions require special attention in high-value contracts."},
    {"id": "contract-002", "text": "SaaS and Technology Contracts: Software licensing and SaaS agreements frequently involve IP ownership disputes, data processing agreements (DPAs) under GDPR/CCPA, and auto-renewal clauses. Jurisdiction clauses often default to Delaware for US entities. Review data residency requirements carefully for cross-border deployments."},
    {"id": "litigation-001", "text": "Litigation Intake Protocol: Commercial disputes with potential damages above $1M require immediate partner notification and conflict-of-interest screening within 24 hours. Preserve litigation holds on all relevant communications and documents immediately upon notice of potential litigation. Statute of limitations deadlines must be docketed on intake."},
    {"id": "litigation-002", "text": "Employment Litigation: Claims involving discrimination, wrongful termination, or wage disputes are subject to EEOC filing requirements and strict administrative deadlines. Class action risk must be assessed at intake. Document retention policies must be immediately suspended for relevant HR records upon receipt of a demand letter."},
    {"id": "ma-001", "text": "M&A Due Diligence Checklist: Corporate acquisitions require review of articles of incorporation, cap tables, material contracts, IP assignments, pending litigation, regulatory approvals, and employment agreements. HSR Act filing thresholds apply for transactions exceeding $119.5M. Cross-border transactions may require additional regulatory approvals (CFIUS, EU Merger Regulation)."},
    {"id": "employment-001", "text": "Executive Compensation and Severance: Non-compete agreements are enforceable in most US jurisdictions but void in California, Minnesota, and North Dakota. Severance agreements for employees over 40 must comply with the ADEA 21/7-day rule. Equity acceleration provisions in change-of-control scenarios require careful tax analysis."},
    {"id": "ip-001", "text": "Intellectual Property Matters: Patent infringement claims require immediate assessment of claim charts and invalidity arguments. Trade secret misappropriation under the DTSA allows for ex parte seizure orders in urgent cases. Trademark clearance searches should be conducted before any new brand launch. Copyright registration strengthens enforcement rights and enables statutory damages claims."},
    {"id": "compliance-001", "text": "Regulatory Compliance: SEC reporting obligations for public companies include 8-K filings for material events within 4 business days. FCPA violations carry both civil and criminal penalties. Financial institutions face BSA/AML compliance requirements with mandatory SAR filings. Data breach notification requirements vary by state but most require notification within 30-72 hours."},
    {"id": "jurisdiction-001", "text": "Delaware Corporate Law: Delaware is the preferred jurisdiction for US corporate matters due to its well-developed case law and business-friendly Court of Chancery. Delaware LLCs offer flexible governance structures. Fiduciary duty standards (Revlon, Unocal, entire fairness) apply in M&A transactions involving Delaware entities. Books and records inspection rights under DGCL Section 220 are frequently litigated."},
    {"id": "jurisdiction-002", "text": "Cross-Border Considerations: Matters involving EU parties require GDPR compliance review. UK post-Brexit matters may require dual compliance with UK GDPR and EU GDPR. Australian corporate matters are governed by the Corporations Act 2001 (Cth). Choice-of-law clauses and forum selection clauses should be reviewed carefully in international commercial contracts."},
]


def seed() -> None:
    added = add_documents(DOCUMENTS)
    total = document_count()
    if added == 0:
        print(f"Knowledge base already seeded ({total} documents). Nothing to add.")
    else:
        print(f"Seeded {added} documents. Total: {total}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed()
