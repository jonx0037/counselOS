"use client";

import { useState } from "react";
import IntakeForm from "@/components/IntakeForm";
import IntakeResultView from "@/components/IntakeResult";
import type { IntakeResult } from "@/types";

export default function HomePage() {
  const [result, setResult] = useState<IntakeResult | null>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-3xl font-semibold text-counsel-navy mb-2">
          Legal Matter Intake
        </h1>
        <p className="text-slate-500 max-w-2xl">
          CounselOS runs your matter submission through a five-agent AI pipeline —
          intake normalization, classification, RAG-augmented context retrieval,
          risk assessment, and structured attorney assignment recommendation.
        </p>
      </div>

      {loading && (
        <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-10 flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-counsel-light border-t-counsel-mid rounded-full animate-spin" />
          <div className="text-center">
            <p className="text-counsel-navy font-medium">Running intake pipeline...</p>
            <p className="text-sm text-slate-400 mt-1">
              Intake → Classification → RAG → Risk → Response
            </p>
          </div>
        </div>
      )}

      {!loading && !result && (
        <IntakeForm
          onResult={setResult}
          onLoading={setLoading}
        />
      )}

      {!loading && result && (
        <IntakeResultView
          result={result}
          onReset={() => setResult(null)}
        />
      )}
    </div>
  );
}
