"use client";

import { useState } from "react";
import type { MatterSubmission, IntakeResult } from "@/types";

interface Props {
  onResult: (result: IntakeResult) => void;
  onLoading: (loading: boolean) => void;
}

export default function IntakeForm({ onResult, onLoading }: Props) {
  const [form, setForm] = useState<MatterSubmission>({
    client_name: "",
    submitted_by: "",
    matter_description: "",
    jurisdiction: "",
    deadline: "",
  });
  const [error, setError] = useState<string | null>(null);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setForm((prev) => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async () => {
    setError(null);
    if (!form.client_name || !form.submitted_by || !form.matter_description) {
      setError("Client name, contact, and matter description are required.");
      return;
    }
    onLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/intake`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Unknown error");
      }
      const data: IntakeResult = await res.json();
      onResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      onLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-8">
      <h2 className="text-xl font-semibold text-counsel-navy mb-1">
        New Matter Submission
      </h2>
      <p className="text-sm text-slate-500 mb-6">
        Submit a legal matter for AI-powered intake, classification, and
        attorney assignment recommendation.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <Field label="Client / Company Name *" name="client_name" value={form.client_name} onChange={handleChange} placeholder="Acme Corporation" />
        <Field label="Submitted By *" name="submitted_by" value={form.submitted_by} onChange={handleChange} placeholder="Jane Smith, VP Legal" />
        <Field label="Jurisdiction" name="jurisdiction" value={form.jurisdiction || ""} onChange={handleChange} placeholder="e.g. Delaware, New York, Federal" />
        <Field label="Deadline / Urgency Note" name="deadline" value={form.deadline || ""} onChange={handleChange} placeholder="e.g. Response required by March 14" />
      </div>

      <div className="mt-5">
        <label className="block text-sm font-medium text-slate-700 mb-1.5">
          Matter Description *
        </label>
        <textarea
          name="matter_description"
          value={form.matter_description}
          onChange={handleChange}
          rows={6}
          placeholder="Describe the legal matter in detail. Include relevant parties, instruments, dates, and context..."
          className="w-full rounded-lg border border-slate-300 px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-counsel-mid focus:border-transparent resize-none"
        />
      </div>

      {error && (
        <div className="mt-4 rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="mt-6 flex justify-end">
        <button
          onClick={handleSubmit}
          className="bg-counsel-blue hover:bg-counsel-navy text-white font-medium px-6 py-2.5 rounded-lg text-sm transition-colors duration-150"
        >
          Run Intake Pipeline →
        </button>
      </div>
    </div>
  );
}

function Field({
  label,
  name,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  name: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  placeholder?: string;
}) {
  return (
    <div>
      <label className="block text-sm font-medium text-slate-700 mb-1.5">
        {label}
      </label>
      <input
        type="text"
        name={name}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className="w-full rounded-lg border border-slate-300 px-4 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-counsel-mid focus:border-transparent"
      />
    </div>
  );
}
