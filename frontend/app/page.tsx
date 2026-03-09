"use client";

import { useState } from "react";
import IntakeForm from "@/components/IntakeForm";
import IntakeResultView from "@/components/IntakeResult";
import ChatPanel from "@/components/ChatPanel";
import type { IntakeResult } from "@/types";

export default function HomePage() {
  const [result, setResult] = useState<IntakeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);

  const handleReset = () => {
    setResult(null);
    setChatOpen(false);
  };

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
        <IntakeForm onResult={setResult} onLoading={setLoading} />
      )}

      {!loading && result && (
        <IntakeResultView result={result} onReset={handleReset} />
      )}

      {/* Chat button — visible when results are showing */}
      {!loading && result && !chatOpen && (
        <button
          onClick={() => setChatOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 bg-counsel-blue text-white rounded-full shadow-lg hover:bg-counsel-navy transition-colors flex items-center justify-center z-40"
          title="Ask about this matter"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </button>
      )}

      {/* Chat panel */}
      {result && (
        <ChatPanel
          result={result}
          open={chatOpen}
          onClose={() => setChatOpen(false)}
        />
      )}
    </div>
  );
}
