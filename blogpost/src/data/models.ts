// One Plotly marker per model, and the same shape as an SVG path for the HTML legend
// (drawn in a 13×13 box centered on the origin).
export const SYMBOLS: Record<string, string> = {
  "nemotron-3.5-lightning": "cross",
  "gpt-oss-120b": "triangle-up",
  "minimax-m2.7": "triangle-down",
  "gpt-5.4-mini": "hexagon",
  "qwen3.6-35b-a3b": "triangle-left",
  "gemma-4-31b-it": "x",
  "mimo-v2.5": "square",
  "deepseek-v4-flash": "triangle-right",
  "gpt-5.6-luna": "pentagon",
};

const CROSS = "M-1.8,-5.5h3.6v3.7h3.7v3.6h-3.7v3.7h-3.6v-3.7h-3.7v-3.6h3.7z";

export const SHAPES: Record<string, { d: string; rotate?: number }> = {
  cross: { d: CROSS },
  x: { d: CROSS, rotate: 45 },
  "triangle-up": { d: "M0,-5.5L5.5,4.5H-5.5Z" },
  "triangle-down": { d: "M0,5.5L5.5,-4.5H-5.5Z" },
  "triangle-left": { d: "M-5.5,0L4.5,-5.5V5.5Z" },
  "triangle-right": { d: "M5.5,0L-4.5,-5.5V5.5Z" },
  hexagon: { d: "M0,-5.5L4.76,-2.75V2.75L0,5.5L-4.76,2.75V-2.75Z" },
  square: { d: "M-4.5,-4.5h9v9h-9z" },
  pentagon: { d: "M0,-5.5L5.23,-1.7L3.23,4.45H-3.23L-5.23,-1.7Z" },
};

// The legend lists models by their MSR on AppWorld test_normal, lowest first.
export const LEGEND_ORDER = [
  "nemotron-3.5-lightning", "gpt-oss-120b", "minimax-m2.7",
  "gpt-5.4-mini", "qwen3.6-35b-a3b", "gemma-4-31b-it",
  "mimo-v2.5", "deepseek-v4-flash", "gpt-5.6-luna",
];
