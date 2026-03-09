"use client";

import type { IntakeResult } from "@/types";

interface Props {
  result: IntakeResult;
  onSelect: (question: string) => void;
}

function generateChips(result: IntakeResult): string[] {
  const chips: string[] = ["What similar matters have we handled?"];

  const candidates: [boolean, string][] = [
    [result.risk_score >= 7, `Why is my risk score ${result.risk_score.toFixed(1)}/10?`],
    [
      result.risk_score >= 5 && result.risk_score < 7,
      "What's driving my risk assessment?",
    ],
    [
      result.risk_flags.conflict_of_interest,
      "Explain the conflict of interest flag",
    ],
    [
      result.risk_flags.jurisdiction_complexity,
      "Explain the jurisdiction complexity flag",
    ],
    [
      result.risk_flags.regulatory_exposure,
      "Explain the regulatory exposure flag",
    ],
    [
      result.risk_flags.time_sensitivity,
      "Explain the time sensitivity flag",
    ],
    [
      result.matter_type_confidence < 0.8,
      `Why was this classified as ${result.matter_type.replace(/_/g, " ")}?`,
    ],
    [
      result.recommended_tier === "partner" ||
        result.recommended_tier === "senior_associate",
      `Why does this need a ${result.recommended_tier === "partner" ? "partner" : "senior associate"}?`,
    ],
    [
      result.urgency === "critical" || result.urgency === "high",
      `What makes this ${result.urgency} urgency?`,
    ],
  ];

  for (const [condition, chip] of candidates) {
    if (chips.length >= 4) break;
    if (condition) chips.push(chip);
  }

  return chips;
}

export default function SuggestedChips({ result, onSelect }: Props) {
  const chips = generateChips(result);

  return (
    <div className="flex flex-wrap gap-2">
      {chips.map((chip) => (
        <button
          key={chip}
          onClick={() => onSelect(chip)}
          className="text-xs bg-counsel-light text-counsel-blue px-3 py-1.5 rounded-full hover:bg-counsel-mid hover:text-white transition-colors"
        >
          {chip}
        </button>
      ))}
    </div>
  );
}
