"use client";

import { useState, useEffect, useRef } from "react";

const ACCENT = "#8B5CF6";

function useReveal() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) { el.classList.add("visible"); obs.disconnect(); } },
      { threshold: 0.12 }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, []);
  return ref;
}

function RevealDiv({ children, className = "", style }: { children: React.ReactNode; className?: string; style?: React.CSSProperties }) {
  const ref = useReveal();
  return <div ref={ref} className={`reveal ${className}`} style={style}>{children}</div>;
}

export default function EvalForgePage() {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (email) setSubmitted(true);
  };

  return (
    <div className="min-h-screen bg-white text-[#0a0a0a]">
      {/* Sticky Nav */}
      <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur border-b border-gray-100">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <a
            href="https://specialized-model-startups.vercel.app"
            className="text-sm text-gray-500 hover:text-gray-900 transition-colors flex items-center gap-1.5"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M9 2L4 7l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            Specialist AI
          </a>
          <span className="font-semibold text-sm tracking-tight">EvalForge</span>
          <a
            href="https://github.com/calebnewtonusc/evalforge"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-gray-500 hover:text-gray-900 transition-colors flex items-center gap-1.5"
          >
            GitHub
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M5 2h7v7M12 2L2 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </a>
        </div>
      </nav>

      {/* Hero */}
      <section id="hero" className="max-w-5xl mx-auto px-6 pt-24 pb-20">
        <div className="animate-fade-up delay-0">
          <div
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium mb-8 border"
            style={{ borderColor: `${ACCENT}40`, color: ACCENT, backgroundColor: `${ACCENT}08` }}
          >
            <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: ACCENT }} />
            Training · ETA Q4 2026
          </div>
        </div>

        <h1
          className="serif animate-fade-up delay-1 text-5xl md:text-7xl font-light leading-[1.05] tracking-tight mb-6"
          style={{ animationDelay: "0.1s" }}
        >
          Evaluations that
          <br />
          <span style={{ color: ACCENT }}>models can&apos;t game.</span>
        </h1>

        <p
          className="animate-fade-up text-lg md:text-xl text-gray-500 max-w-2xl leading-relaxed mb-4"
          style={{ animationDelay: "0.2s" }}
        >
          AI evaluation intelligence for ML researchers and teams.
        </p>
        <p
          className="animate-fade-up text-base text-gray-400 max-w-2xl leading-relaxed mb-12"
          style={{ animationDelay: "0.25s" }}
        >
          The only model trained on evaluation quality — detecting shortcuts, contamination, and Goodhart failures before they corrupt your benchmark scores.
        </p>

        {!submitted ? (
          <form
            onSubmit={handleSubmit}
            className="animate-fade-up flex flex-col sm:flex-row gap-3 max-w-md"
            style={{ animationDelay: "0.3s" }}
          >
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="your@email.com"
              required
              className="flex-1 px-4 py-2.5 rounded-lg border border-gray-200 text-sm focus:outline-none focus:border-purple-300 transition-colors bg-white"
            />
            <button
              type="submit"
              className="px-5 py-2.5 rounded-lg text-sm font-medium text-white transition-opacity hover:opacity-90"
              style={{ backgroundColor: ACCENT }}
            >
              Join Waitlist
            </button>
          </form>
        ) : (
          <div
            className="animate-fade-up text-sm font-medium px-4 py-2.5 rounded-lg inline-block"
            style={{ color: ACCENT, backgroundColor: `${ACCENT}10`, border: `1px solid ${ACCENT}30` }}
          >
            You are on the list. We will reach out before launch.
          </div>
        )}
      </section>

      {/* The Gap */}
      <section id="gap" className="border-t border-gray-100 bg-gray-50/50">
        <div className="max-w-5xl mx-auto px-6 py-20">
          <RevealDiv className="mb-12">
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-2">The Gap</p>
            <h2 className="serif text-3xl md:text-4xl font-light tracking-tight">
              What changes when a model specializes
            </h2>
          </RevealDiv>
          <div className="grid md:grid-cols-2 gap-6">
            <RevealDiv className="bg-white border border-gray-200 rounded-xl p-6">
              <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-4">General Models</p>
              <p className="text-gray-700 leading-relaxed text-sm">
                MMLU saturated at 90%. Models score high by memorizing. Goodhart&apos;s Law wins: the measure becomes the target. Benchmark performance stops predicting real capability.
              </p>
              <ul className="mt-4 space-y-2">
                {["Score high by memorizing benchmarks", "Cannot detect evaluation shortcuts", "Contamination goes undetected", "Goodhart failure corrupts downstream decisions"].map((item) => (
                  <li key={item} className="flex items-start gap-2 text-sm text-gray-500">
                    <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <circle cx="7" cy="7" r="6" stroke="#E5E7EB" strokeWidth="1.5"/>
                      <path d="M4.5 7l1.5 1.5 3-3" stroke="#9CA3AF" strokeWidth="1.2" strokeLinecap="round"/>
                    </svg>
                    {item}
                  </li>
                ))}
              </ul>
            </RevealDiv>
            <RevealDiv className="border rounded-xl p-6" style={{ borderColor: `${ACCENT}30`, backgroundColor: `${ACCENT}04` }}>
              <p className="text-xs font-semibold uppercase tracking-widest mb-4" style={{ color: ACCENT }}>EvalForge</p>
              <p className="text-gray-700 leading-relaxed text-sm">
                EvalForge continuously generates items models haven&apos;t seen, probes for shortcuts, and tracks whether your eval still predicts real capability.
              </p>
              <ul className="mt-4 space-y-2">
                {["Contamination probes for any benchmark", "Shortcut detection — annotation artifacts, length biases", "New evaluation items resistant to memorization", "Correlation tracking: benchmark score → real task performance"].map((item) => (
                  <li key={item} className="flex items-start gap-2 text-sm text-gray-700">
                    <svg className="mt-0.5 shrink-0" width="14" height="14" viewBox="0 0 14 14" fill="none">
                      <circle cx="7" cy="7" r="6" stroke={ACCENT} strokeWidth="1.5" fill={`${ACCENT}15`}/>
                      <path d="M4.5 7l1.5 1.5 3-3" stroke={ACCENT} strokeWidth="1.2" strokeLinecap="round"/>
                    </svg>
                    {item}
                  </li>
                ))}
              </ul>
            </RevealDiv>
          </div>
        </div>
      </section>

      {/* How It's Built */}
      <section id="how" className="max-w-5xl mx-auto px-6 py-20">
        <RevealDiv className="mb-12">
          <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-2">How It&apos;s Built</p>
          <h2 className="serif text-3xl md:text-4xl font-light tracking-tight">
            Three-stage training pipeline
          </h2>
        </RevealDiv>
        <div className="grid md:grid-cols-3 gap-5">
          {[
            {
              stage: "Stage 1",
              name: "Supervised Fine-Tuning",
              desc: "Train on 2M+ reviews from OpenReview + BIG-Bench/HELM/MMLU critique papers + shortcut studies. Model learns the anatomy of benchmark failure.",
            },
            {
              stage: "Stage 2",
              name: "Reinforcement Learning",
              desc: "Reward signal: shortcut detection accuracy + downstream task correlation coefficient stability. RL teaches the model to generate evals that actually predict capability.",
            },
            {
              stage: "Stage 3",
              name: "Direct Preference Optimization",
              desc: "DPO on pairs where generated items successfully probed shortcut reliance versus items that failed. Calibrates probe quality and item diversity.",
            },
          ].map((s, i) => (
            <RevealDiv key={s.stage} className="border border-gray-200 rounded-xl p-6 flex flex-col gap-3" style={{ animationDelay: `${i * 0.1}s` }}>
              <div className="flex items-center gap-2">
                <span
                  className="text-xs font-bold px-2 py-0.5 rounded"
                  style={{ color: ACCENT, backgroundColor: `${ACCENT}12` }}
                >
                  {s.stage}
                </span>
              </div>
              <p className="font-semibold text-sm">{s.name}</p>
              <p className="text-sm text-gray-500 leading-relaxed">{s.desc}</p>
            </RevealDiv>
          ))}
        </div>
      </section>

      {/* Capabilities */}
      <section id="capabilities" className="border-t border-gray-100 bg-gray-50/50">
        <div className="max-w-5xl mx-auto px-6 py-20">
          <RevealDiv className="mb-12">
            <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-2">Capabilities</p>
            <h2 className="serif text-3xl md:text-4xl font-light tracking-tight">What it can do</h2>
          </RevealDiv>
          <div className="grid md:grid-cols-2 gap-5">
            {[
              {
                icon: (
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <circle cx="10" cy="10" r="7" stroke={ACCENT} strokeWidth="1.5"/>
                    <path d="M7 10l2 2 4-4" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ),
                title: "Contamination Probe Generation",
                desc: "Generates membership inference probes to detect whether a model has seen benchmark items in training — for any benchmark.",
              },
              {
                icon: (
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <path d="M4 16l4-8 4 4 4-8" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                ),
                title: "Shortcut Detection",
                desc: "Finds annotation artifacts, length biases, and spurious correlations that let models score high without understanding the task.",
              },
              {
                icon: (
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <rect x="3" y="3" width="14" height="14" rx="2" stroke={ACCENT} strokeWidth="1.5"/>
                    <path d="M7 10h6M10 7v6" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                ),
                title: "New Evaluation Item Generation",
                desc: "Generates novel evaluation items resistant to memorization — keeping your benchmark valid as models scale and data pipelines expand.",
              },
              {
                icon: (
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                    <path d="M5 15l3-6 3 4 2-3 2 5" stroke={ACCENT} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <rect x="2" y="2" width="16" height="16" rx="2" stroke={ACCENT} strokeWidth="1.5"/>
                  </svg>
                ),
                title: "Correlation Tracking",
                desc: "Tracks whether benchmark score still predicts downstream real-task performance — alerting when an eval has Goodhart-failed.",
              },
            ].map((cap) => (
              <RevealDiv key={cap.title} className="bg-white border border-gray-200 rounded-xl p-6 flex gap-4">
                <div className="shrink-0 mt-0.5">{cap.icon}</div>
                <div>
                  <p className="font-semibold text-sm mb-1.5">{cap.title}</p>
                  <p className="text-sm text-gray-500 leading-relaxed">{cap.desc}</p>
                </div>
              </RevealDiv>
            ))}
          </div>
        </div>
      </section>

      {/* Training Stats */}
      <section id="stats" className="max-w-5xl mx-auto px-6 py-20">
        <RevealDiv className="mb-12">
          <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-2">Training</p>
          <h2 className="serif text-3xl md:text-4xl font-light tracking-tight">The numbers behind the model</h2>
        </RevealDiv>
        <div className="grid md:grid-cols-3 gap-5">
          {[
            { label: "Dataset", value: "2M+", sub: "reviews from OpenReview + BIG-Bench/HELM/MMLU critique papers + shortcut studies" },
            { label: "Base Model", value: "Qwen2.5", sub: "7B-Coder-Instruct — specialized code foundation" },
            { label: "Reward Signal", value: "Dual", sub: "Shortcut detection accuracy + downstream task correlation coefficient stability" },
          ].map((stat) => (
            <RevealDiv
              key={stat.label}
              className="rounded-xl p-6 border"
              style={{ borderColor: `${ACCENT}25`, backgroundColor: `${ACCENT}05` }}
            >
              <p className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-2">{stat.label}</p>
              <p className="text-2xl font-bold mb-1" style={{ color: ACCENT }}>{stat.value}</p>
              <p className="text-sm text-gray-500 leading-relaxed">{stat.sub}</p>
            </RevealDiv>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-100">
        <div className="max-w-5xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p className="text-xs text-gray-400">
            Part of the{" "}
            <a href="https://specialized-model-startups.vercel.app" className="underline underline-offset-2 hover:text-gray-600 transition-colors">
              Specialist AI
            </a>{" "}
            portfolio
          </p>
          <p className="text-xs text-gray-400">Caleb Newton · USC · 2026</p>
        </div>
      </footer>
    </div>
  );
}
