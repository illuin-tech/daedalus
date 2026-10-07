// The paper's tables (daedalus-preprint.pdf, main body), as numbers. `se` is the standard error
// over five inference runs.

export type V = { v: number; se?: number };
const v = (x: number, se?: number): V => ({ v: x, se });

// ── Table 1: main results ────────────────────────────────────────────────
export const BENCHES = [
  { id: "appworld", name: "AppWorld", split: "normal" },
  { id: "tau2", name: "τ²-bench", split: "retail" },
  { id: "ab", name: "AutomationBench", split: "Operations" },
] as const;

/** Each baseline's paper (arXiv abstract page). */
export const PAPERS: Record<string, string> = {
  AutoGuide: "https://arxiv.org/abs/2403.08978",
  ReasoningBank: "https://arxiv.org/abs/2509.25140",
  ERL: "https://arxiv.org/abs/2603.24639",
  ExpeL: "https://arxiv.org/abs/2308.10144",
  ACE: "https://arxiv.org/abs/2510.04618",
  PREPING: "https://arxiv.org/abs/2605.13880",
};

export type Row = { name: string; group?: "training" | "none"; ours?: boolean; cells: Record<string, [V, V, V]> };

export const TABLE1: Row[] = [
  { name: "No-memory baseline", cells: {
    appworld: [v(44.3, 1.1), v(14.9, 2.8), v(3.2)], tau2: [v(57.5, 1.6), v(22.5, 6.6), v(0.5)], ab: [v(31.7, 3.2), v(12.9, 4.0), v(1.0)] } },
  { name: "AutoGuide", group: "training", cells: {
    appworld: [v(44.0, 0.5), v(17.9, 3.0), v(7.6)], tau2: [v(61.5, 2.3), v(27.5, 7.1), v(3.2)], ab: [v(35.4, 1.7), v(18.6, 4.6), v(4.7)] } },
  { name: "ReasoningBank", group: "training", cells: {
    appworld: [v(48.0, 1.0), v(19.6, 3.1), v(3.3)], tau2: [v(64.0, 2.9), v(32.5, 7.5), v(0.5)], ab: [v(19.1, 1.7), v(2.9, 2.0), v(0.7)] } },
  { name: "ERL", group: "training", cells: {
    appworld: [v(58.6, 1.3), v(29.2, 3.5), v(23.0)], tau2: [v(64.0, 3.6), v(30.0, 7.2), v(5.1)], ab: [v(33.4, 1.5), v(15.7, 4.3), v(3.2)] } },
  { name: "ExpeL", group: "training", cells: {
    appworld: [v(59.0, 1.9), v(33.3, 3.6), v(4.6)], tau2: [v(67.0, 1.8), v(37.5, 7.8), v(0.9)], ab: [v(42.6, 0.5), v(21.4, 4.9), v(1.5)] } },
  { name: "ACE", group: "training", cells: {
    appworld: [v(60.5, 0.8), v(41.1, 3.8), v(7.6)], tau2: [v(70.5, 3.8), v(37.5, 7.8), v(2.2)], ab: [v(26.3, 1.9), v(10.0, 3.6), v(0.8)] } },
  { name: "DAEDALUS-curated", group: "training", cells: {
    appworld: [v(60.8, 0.9), v(36.3, 3.7), v(3.0)], tau2: [v(70.0, 3.6), v(35.0, 7.5), v(0.6)], ab: [v(40.0, 2.3), v(24.3, 5.1), v(1.0)] } },
  { name: "PREPING", group: "none", cells: {
    appworld: [v(56.0, 1.4), v(25.6, 3.4), v(4.0)], tau2: [v(60.0, 1.8), v(25.0, 6.9), v(0.7)], ab: [v(34.9, 1.5), v(15.7, 4.3), v(1.1)] } },
  { name: "DAEDALUS", group: "none", ours: true, cells: {
    appworld: [v(60.2, 0.9), v(32.1, 3.6), v(3.2)], tau2: [v(67.5, 2.5), v(37.5, 7.8), v(0.5)], ab: [v(36.0, 1.5), v(21.4, 4.9), v(1.2)] } },
];

// ── Table 2: cross-family transfer (AppWorld MSR) ────────────────────────
export const AGENTS = ["GPT-5.4-mini", "Qwen3.6-35B-A3B", "DeepSeek-V4-Flash"];
export const TRANSFER = {
  baseline: [v(44.3, 1.1), v(44.8, 1.3), v(80.6, 1.6)],
  banks: [
    { aux: "GPT-5.4", solver: "GPT-5.4-mini", family: 0, gains: [v(15.9, 1.4), v(16.1, 1.6), v(8.3, 1.8)] },
    { aux: "Qwen3.8-Flash", solver: "Qwen3.6-35B-A3B", family: 1, gains: [v(16.8, 2.4), v(29.3, 1.7), v(4.9, 2.0)] },
    { aux: "DeepSeek-V4-Pro", solver: "DeepSeek-V4-Flash", family: 2, gains: [v(11.4, 3.6), v(14.5, 1.3), v(3.0, 3.0)] },
  ],
};

// ── Table 3: cumulative ablation of the generation pipeline (AppWorld) ───
export type AblationRow = {
  id: string; name: string; msr: V; pass5: V; inf: number; accept?: number; refin?: number; gen?: number;
  /** Diagram parts this row adds (cumulative), and the arrows it brings in. */
  adds: string[]; arrows: string[]; note: string;
};

export const ABLATION_BASELINE = { msr: v(44.3, 1.1), pass5: v(14.9, 2.8), inf: 3.2 };

export const ABLATION: AblationRow[] = [
  { id: "A", name: "Single Explorer", msr: v(35.7, 0.8), pass5: v(10.1, 2.3), inf: 3.5, gen: 5.2,
    adds: ["explorer", "hbank", "consolidator", "memory"], arrows: ["a-consolidate", "a-deploy"],
    note: "Heuristics drawn from one 100-turn Explorer trajectory. Exploring alone <b>hurts</b>: the agent does worse than with no memory at all." },
  { id: "B", name: "+ Multi-session Explorer", msr: v(38.7, 3.6), pass5: v(11.3, 2.4), inf: 2.9, gen: 92.0,
    adds: ["sessions"], arrows: [],
    note: "90 sessions of 40 turns, each seeing the heuristics accepted so far. Eighteen times the cost of (A), still below the baseline." },
  { id: "C", name: "+ Explorer tasks and Solver traces", msr: v(54.5, 2.8), pass5: v(25.0, 3.3), inf: 1.9, gen: 46.6,
    adds: ["loop", "solver", "extractor"], arrows: ["a-task", "a-heur"],
    note: "The Explorer now proposes tasks, and a heuristic is drawn from every Solver trajectory. <b>+15.8 MSR</b>: what is worth remembering lies in the agent's own attempts." },
  { id: "D", name: "+ Solver loop", msr: v(59.0, 1.4), pass5: v(30.4, 3.6), inf: 3.3, accept: 86.7, refin: 183, gen: 210.4,
    adds: ["judge"], arrows: ["a-cond", "a-trace", "a-fail", "a-bank", "a-feedback"],
    note: "Heuristics are revised after each failure and kept only after three successes in a row. <b>+4.5 MSR</b>: validation removes the noisy lessons." },
  { id: "E", name: "+ Explorer guideline memory", msr: v(57.9, 2.1), pass5: v(31.5, 3.6), inf: 3.2, accept: 91.1, refin: 161, gen: 185.3,
    adds: ["guidelines"], arrows: ["a-guide", "a-lesson"],
    note: "Refinement lessons carried into later sessions: fewer refinements (183 → 161) and 12% cheaper, with MSR within error." },
  { id: "F", name: "+ Environment survey (DAEDALUS)", msr: v(60.2, 0.9), pass5: v(32.1, 3.6), inf: 3.2, accept: 97.8, refin: 118, gen: 109.7,
    adds: ["surveyor"], arrows: ["a-tags"],
    note: "A target distribution of tasks over the environment: 97.8% of sessions end with an accepted heuristic, and generation costs <b>48% less</b> than (D)." },
];

// ── Table 4: the LLM judge against the official verifiers ────────────────
export const JUDGE = [
  { bench: "AppWorld", n: 168, precision: 0.943, recall: 0.846, kbench: 0.807, kinter: 0.848 },
  { bench: "τ²-bench", n: 40, precision: 0.889, recall: 0.842, kbench: 0.749, kinter: 1.0 },
  { bench: "AutomationBench", n: 70, precision: 0.863, recall: 0.807, kbench: 0.729, kinter: 0.902 },
];

// ── Table 5: auxiliary model ─────────────────────────────────────────────
export const AUX = [
  { model: "no memory", msr: v(44.3, 1.1), pass5: v(14.9, 2.8), gen: 0 },
  { model: "GPT-5.4-mini", msr: v(51.2, 1.6), pass5: v(20.2, 3.1), gen: 67.2 },
  { model: "GPT-5.4", msr: v(60.2, 0.9), pass5: v(32.1, 3.6), gen: 109.7, ours: true },
];

// ── Table 6: consolidation and injection (AppWorld) ──────────────────────
export const INJECTION = [
  { group: "", policy: "No memory", dedup: "–", consol: "–", msr: v(44.3, 1.1), pass5: v(14.9, 2.8), cost: 3.2 },
  { group: "Top-5 per turn", policy: "Random", dedup: "✓", consol: "✓", msr: v(51.7, 1.5), pass5: v(19.0, 3.0), cost: 4.1 },
  { group: "Top-5 per turn", policy: "BM25", dedup: "✓", consol: "✓", msr: v(52.0, 1.3), pass5: v(22.0, 3.2), cost: 3.8 },
  { group: "Top-5 per turn", policy: "Qwen3-Emb.", dedup: "✓", consol: "✓", msr: v(54.3, 1.6), pass5: v(24.4, 3.3), cost: 3.7 },
  { group: "Whole memory per turn", policy: "All", dedup: "✓", consol: "✓", msr: v(51.5, 1.2), pass5: v(28.0, 3.5), cost: 3.5 },
  { group: "Whole memory at start", policy: "All", dedup: "✗", consol: "✗", msr: v(49.5, 1.0), pass5: v(26.2, 3.4), cost: 4.9 },
  { group: "Whole memory at start", policy: "All", dedup: "✓", consol: "✗", msr: v(49.8, 2.5), pass5: v(27.4, 3.5), cost: 4.8 },
  { group: "Whole memory at start", policy: "DAEDALUS", dedup: "✓", consol: "✓", msr: v(60.2, 0.9), pass5: v(32.1, 3.6), cost: 3.2, ours: true },
];
