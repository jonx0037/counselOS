"use client";

import type { IntakeResult, UrgencyLevel, AssignmentTier } from "@/types";

const URGENCY_STYLES: Record<UrgencyLevel, string> = {
  critical: "bg-red-100 text-red-800 border-red-200",
  high: "bg-orange-100 text-orange-800 border-orange-200",
  standard: "bg-blue-100 text-blue-800 border-blue-200",
  low: "bg-slate-100 text-slate-600 border-slate-200",
};

const TIER_LABELS: Record<AssignmentTier, string> = {
  partner: "Partner",
  senior_associate: "Senior Associate",
  associate: "Associate",
  paralegal: "Paralegal",
};

const TIER_STYLES: Record<AssignmentTier, string> = {
  partner: "bg-counsel-navy text-white",
  senior_associate: "bg-counsel-blue text-white",
  associate: "bg-counsel-mid text-white",
  paralegal: "bg-slate-500 text-white",
};

interface Props {
  result: IntakeResult;
  onReset: () => void;
}

export default function IntakeResultView({ result, onReset }: Props) {
  const activeFlags = [
    result.risk_flags.conflict_of_interest && "Conflict of Interest",
    result.risk_flags.jurisdiction_complexity && "Jurisdiction Complexity",
    result.risk_flags.regulatory_exposure && "Regulatory Exposure",
    result.risk_flags.time_sensitivity && "Time Sensitivity",
  ].filter(Boolean) as string[];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-counsel-navy rounded-xl p-6 text-white">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <p className="text-counsel-light text-xs font-mono mb-1">{result.matter_id}</p>
            <h2 className="text-2xl font-semibold">{result.client_name}</h2>
            <p className="text-slate-300 text-sm mt-0.5">Submitted by {result.submitted_by}</p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${URGENCY_STYLES[result.urgency]}`}>
              {result.urgency.toUpperCase()}
            </span>
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${TIER_STYLES[result.recommended_tier]}`}>
              → {TIER_LABELS[result.recommended_tier]}
            </span>
          </div>
        </div>
      </div>

      {/* Classification + Risk */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <Card title="Classification">
          <p className="text-sm font-medium text-counsel-blue capitalize mb-1">
            {result.matter_type.replace(/_/g, " ")}
          </p>
          <div className="flex items-center gap-2 mb-3">
            <div className="flex-1 bg-slate-200 rounded-full h-1.5">
              <div
                className="bg-counsel-mid h-1.5 rounded-full"
                style={{ width: `${result.matter_type_confidence * 100}%` }}
              />
            </div>
            <span className="text-xs text-slate-500">
              {Math.round(result.matter_type_confidence * 100)}% confident
            </span>
          </div>
          <p className="text-sm text-slate-600">{result.matter_summary}</p>
        </Card>

        <Card title="Risk Assessment">
          <div className="flex items-center gap-3 mb-3">
            <div className="text-3xl font-bold text-counsel-navy">
              {result.risk_score.toFixed(1)}
            </div>
            <div className="text-xs text-slate-500">/ 10.0<br />risk score</div>
          </div>
          {activeFlags.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {activeFlags.map((flag) => (
                <span key={flag} className="text-xs bg-amber-100 text-amber-800 border border-amber-200 rounded px-2 py-0.5">
                  ⚠ {flag}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-400 italic">No flags raised</p>
          )}
          {result.risk_flags.notes.length > 0 && (
            <ul className="mt-3 space-y-1">
              {result.risk_flags.notes.map((note, i) => (
                <li key={i} className="text-xs text-slate-600">• {note}</li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      {/* Intake Summary */}
      <Card title="Intake Summary">
        <p className="text-sm text-slate-700 leading-relaxed">{result.intake_summary}</p>
      </Card>

      {/* Next Steps */}
      <Card title="Suggested Next Steps">
        <ol className="space-y-2">
          {result.suggested_next_steps.map((step, i) => (
            <li key={i} className="flex gap-3 text-sm text-slate-700">
              <span className="flex-shrink-0 w-5 h-5 rounded-full bg-counsel-light text-counsel-blue text-xs font-semibold flex items-center justify-center mt-0.5">
                {i + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
      </Card>

      {/* Pipeline Metadata */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 text-xs text-slate-500 font-mono">
        <div className="flex flex-wrap gap-4">
          <span>agents: {result.agents_run.join(" → ")}</span>
          <span>tokens: {result.total_tokens_used.toLocaleString()}</span>
          <span>model: {result.llm_model}</span>
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={onReset}
          className="text-sm text-counsel-mid hover:text-counsel-navy underline"
        >
          ← Submit another matter
        </button>
      </div>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
      <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wide mb-3">
        {title}
      </h3>
      {children}
    </div>
  );
}
