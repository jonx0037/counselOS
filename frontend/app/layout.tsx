import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CounselOS — Legal Matter Intake",
  description: "AI-powered corporate legal matter intake and triage system",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 font-sans antialiased">
        <header className="bg-counsel-navy border-b border-counsel-blue">
          <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded bg-counsel-gold flex items-center justify-center">
                <span className="text-counsel-navy font-bold text-sm">⚖</span>
              </div>
              <div>
                <span className="text-white font-semibold text-lg tracking-tight">CounselOS</span>
                <span className="text-counsel-light text-xs ml-2 opacity-70">
                  Legal Matter Intake
                </span>
              </div>
            </div>
            <span className="text-xs text-slate-400 font-mono">AI Co-Worker · Demo</span>
          </div>
        </header>
        <main className="max-w-5xl mx-auto px-6 py-10">{children}</main>
        <footer className="text-center text-xs text-slate-400 pb-8">
          CounselOS · Built by{" "}
          <a href="https://datasalt.ai" className="underline hover:text-counsel-mid">
            DataSalt LLC
          </a>
        </footer>
      </body>
    </html>
  );
}
