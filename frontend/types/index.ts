export type MatterType =
  | "contract"
  | "litigation"
  | "mergers_acquisitions"
  | "employment"
  | "intellectual_property"
  | "compliance"
  | "real_estate"
  | "corporate_governance"
  | "unknown";

export type UrgencyLevel = "critical" | "high" | "standard" | "low";
export type AssignmentTier = "partner" | "senior_associate" | "associate" | "paralegal";

export interface MatterSubmission {
  client_name: string;
  submitted_by: string;
  matter_description: string;
  jurisdiction?: string;
  deadline?: string;
}

export interface RiskFlags {
  conflict_of_interest: boolean;
  jurisdiction_complexity: boolean;
  regulatory_exposure: boolean;
  time_sensitivity: boolean;
  notes: string[];
}

export interface IntakeResult {
  matter_id: string;
  client_name: string;
  submitted_by: string;
  submitted_at: string;
  matter_type: MatterType;
  matter_type_confidence: number;
  matter_summary: string;
  retrieved_context: string[];
  urgency: UrgencyLevel;
  risk_score: number;
  risk_flags: RiskFlags;
  recommended_tier: AssignmentTier;
  intake_summary: string;
  suggested_next_steps: string[];
  agents_run: string[];
  total_tokens_used: number;
  llm_model: string;
}
