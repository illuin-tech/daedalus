// The paper's other main-body figures, redrawn: model families (Figure 1), generation cost by
// role (Figure 5), the exploration budget (Figure 6) and per-turn retrieval (Figure 7).
import families from "../data/fig-transfer.json";
import genCost from "../data/fig-gen-cost.json";
import sessions from "../data/fig-sessions.json";
import retrieval from "../data/fig-retrieval.json";
import { axis, baseLayout, drawLinesAt, hexA, mix, mount, scrub, type Tokens } from "./chart-theme";
import { clamp01 } from "./scroll";

const byId = (id: string) => document.getElementById(id);

// ── model families: each model moves from its no-memory point to its point with memory, ──
// following the scroll (p = 0: no memory, p = 1: with memory; `scrub` in chart-theme.ts).
const GLYPH: Record<string, string> = {
  v: "triangle-down", s: "square", D: "diamond", "^": "triangle-up", o: "circle", p: "pentagon", h: "hexagon",
};

function familyFrame(p: number) {
  return (t: Tokens, narrow: boolean) => {
    const data = families.models.flatMap((m) => {
      const b = m.base;
      const e = { turns: b.turns + (m.memory.turns - b.turns) * p, msr: b.msr + (m.memory.msr - b.msr) * p };
      const symbol = GLYPH[m.glyph] ?? "circle";
      return [
        { // the arrow's shaft, from the no-memory point to the current one
          type: "scatter", mode: "lines", x: [b.turns, e.turns], y: [b.msr, e.msr],
          line: { color: hexA(t.orange, 0.35), width: 2 }, hoverinfo: "skip", showlegend: false,
        },
        { // the no-memory point, hollow
          type: "scatter", mode: "markers", x: [b.turns], y: [b.msr],
          marker: { symbol, size: 13, color: t.bg, line: { color: t.muted, width: 1.8 } }, showlegend: false,
          hovertemplate: `<b>${m.id}</b> · no memory<br>MSR %{y:.1f}% · %{x:.1f} turns<extra></extra>`,
        },
        { // the current point: filled once the memory is in
          type: "scatter", mode: "markers+text", x: [e.turns], y: [e.msr], cliponaxis: false,
          text: [narrow ? m.id.replace(/-(flash|lightning|35b-a3b|0731)$/, "") : m.id],
          textposition: "top center", textfont: { size: narrow ? 9 : 11, color: t.muted },
          marker: { symbol, size: 14, color: mix(t.bg, t.teal, p), line: { color: t.ink, width: 1.8 } }, showlegend: false,
          hovertemplate:
            // the final values, whatever the point's position mid-scroll
            `<b>${m.id}</b> · with memory<br>MSR ${m.memory.msr.toFixed(1)}% (${m.delta_msr >= 0 ? "+" : ""}${m.delta_msr.toFixed(1)})` +
            `<br>${m.memory.turns.toFixed(1)} turns (${m.delta_turns.toFixed(1)})<extra></extra>`,
        },
      ];
    });
    // The legend (bottom left): empty traces that only carry the two marker styles.
    const key = (name: string, marker: Record<string, unknown>) =>
      ({ type: "scatter", mode: "markers", x: [null], y: [null], name, marker: { symbol: "circle", size: 11, ...marker }, hoverinfo: "skip" });
    const legend = [
      key("No memory", { color: t.bg, line: { color: t.muted, width: 1.8 } }),
      key("DAEDALUS", { color: t.teal, line: { color: t.ink, width: 1.8 } }),
    ];
    return {
      data: [...data, ...legend],
      layout: baseLayout(t, narrow, {
        showlegend: true,
        legend: {
          x: 0.015, y: 0.02, xanchor: "left", yanchor: "bottom", orientation: "v",
          bgcolor: t.bg, bordercolor: t.ruleStrong, borderwidth: 1,
          font: { size: narrow ? 10.5 : 12, color: t.ink }, itemclick: false, itemdoubleclick: false,
        },
        xaxis: axis(t, families.x.label, { range: [5.5, 22], dtick: 2 }, narrow),
        yaxis: axis(t, families.y.label, { range: [0, 100], dtick: 20, ticksuffix: "%" }, narrow),
      }),
    };
  };
}

const famEl = byId("families-chart");
if (famEl) void scrub(famEl, { frame: familyFrame });

// ── generation cost by role: stacked bars, built layer by layer as the page scrolls ──────
// The roles stack from the bottom up, one after the other (explorer, solver, judge, extractor),
// in all three bars at once; each segment's label fades in as it completes, and each bar's
// total rides on top of it, counting up. Role colours are page tokens (--role-*).

function costFrame(p: number) {
  return (t: Tokens, narrow: boolean) => {
    const x = genCost.bars.map((b) => (narrow ? b.tag : `${b.tag} ${b.label.replace("Daedalus", "DAEDALUS")}`));
    const n = genCost.roles.length;
    const grown = (k: number) => clamp01(p * n - k);  // how far the k-th role's layer has grown
    const cost = (b: (typeof genCost.bars)[number], role: string) => b.roles[role as keyof typeof b.roles];
    const data = genCost.roles.map((role, k) => ({
      type: "bar", name: role, x,
      y: genCost.bars.map((b) => cost(b, role) * grown(k)),
      marker: { color: hexA(t.roles[role], 0.7), line: { color: t.bg, width: 2 } },
      // A label only where the segment is tall enough to hold it.
      text: genCost.bars.map((b) => (cost(b, role) >= 14 ? `$${Math.round(cost(b, role))}` : "")),
      textposition: "inside", insidetextanchor: "middle", constraintext: "none",
      textfont: { size: 11, color: `rgba(21,23,26,${clamp01((grown(k) - 0.6) / 0.4)})` },
      hovertemplate: `%{x}<br>${role}: $%{customdata:.1f}<extra></extra>`,  // the full cost, mid-scroll too
      customdata: genCost.bars.map((b) => cost(b, role)),
    }));
    const annotations = genCost.bars.map((b, i) => {
      const height = genCost.roles.reduce((sum, role, k) => sum + cost(b, role) * grown(k), 0);
      return {
        x: x[i], y: height, yanchor: "bottom", showarrow: false, opacity: clamp01(p / 0.1),
        text: `<b>$${Math.round(height)}</b>`, font: { size: 12.5, color: t.ink },
      };
    });
    return {
      data,
      layout: baseLayout(t, narrow, {
        barmode: "stack", bargap: 0.45,
        xaxis: axis(t, "", { showgrid: false }, narrow),
        yaxis: axis(t, "generation cost ($, one run)", { range: [0, 240], dtick: 50, tickprefix: "$" }, narrow),
        annotations,
      }),
    };
  };
}

const costEl = byId("cost-chart");
if (costEl) void scrub(costEl, { frame: costFrame });

// ── exploration budget: MSR and pass^5 per checkpoint, memory size below ───────
function sessionsChart(t: Tokens, narrow: boolean) {
  const pts = sessions.points;
  const x = pts.map((p) => Math.sqrt(p.sessions));
  const line = (key: "msr" | "pass_k", colour: string, name: string, dash?: string) => ({
    type: "scatter", mode: "lines+markers", name, x,
    y: pts.map((p) => p[key]),
    error_y: { type: "data", array: pts.map((p) => (key === "msr" ? p.msr_se : p.pass_k_se)), color: hexA(colour, 0.5), thickness: 1.2, width: 4 },
    line: { color: colour, width: 2.2, dash },
    marker: { size: 8, color: colour },
    hovertemplate: `${name}: %{y:.1f}%<extra></extra>`,
    xaxis: "x", yaxis: "y",
  });
  const bank = {
    type: "bar", name: "heuristics in memory", x, y: pts.map((p) => p.pool_size ?? 0),
    width: 0.28, marker: { color: hexA(t.muted, 0.35) },
    text: pts.map((p) => (p.pool_size ? String(p.pool_size) : "")), textposition: "outside",
    textfont: { size: 10.5, color: t.muted }, cliponaxis: false,
    hovertemplate: "%{y} heuristics<extra></extra>", xaxis: "x", yaxis: "y2",
  };
  const ticks = { tickvals: x, ticktext: pts.map((p) => (p.sessions === 0 ? "none" : String(p.sessions))) };
  return {
    data: [line("msr", t.teal, "mean success rate"), line("pass_k", t.orange, "pass^5"), bank],
    layout: baseLayout(t, narrow, {
      hovermode: "x unified",
      grid: { rows: 2, columns: 1, subplots: [["xy"], ["xy2"]], roworder: "top to bottom" },
      xaxis: axis(t, "number of sessions", { ...ticks, range: [-0.4, 12.4], anchor: "y2" }, narrow),
      yaxis: axis(t, "success", { domain: [0.34, 1], range: [0, 72], dtick: 20, ticksuffix: "%" }, narrow),
      yaxis2: axis(t, "memory", { domain: [0, 0.22], range: [0, 110], showticklabels: false, dtick: 50 }, narrow),
      annotations: [
        { x: x[5], y: 60.2, text: "peak", showarrow: false, yshift: 16, font: { size: 11, color: t.tealInk } },
      ],
    }),
  };
}

const sesEl = byId("sessions-chart");
if (sesEl) void mount(sesEl, sessionsChart, "draw-lines");

// ── retrieval per turn: BM25 top-k against the whole memory at task start ──────
function retrievalChart(t: Tokens, narrow: boolean) {
  const pts = retrieval.points;
  const x = pts.map((p) => Math.log2(p.k + 1));
  const ref = retrieval.reference;
  const line = (key: "msr" | "pass_k", colour: string, name: string) => ({
    type: "scatter", mode: "lines+markers", name, x, y: pts.map((p) => p[key]),
    line: { color: colour, width: 2.2 }, marker: { size: 8, color: colour },
    hovertemplate: `top-%{customdata} per turn<br>${name}: %{y:.1f}%<extra></extra>`, customdata: pts.map((p) => p.k),
  });
  const hline = (y: number, colour: string) => ({
    type: "scatter", mode: "lines", x: [x[0] - 0.2, x[x.length - 1] + 0.2], y: [y, y],
    line: { color: colour, width: 1.6, dash: "dot" }, hoverinfo: "skip",
  });
  return {
    data: [hline(ref.msr, t.teal), hline(ref.pass_k, t.orange), line("msr", t.teal, "mean success rate"), line("pass_k", t.orange, "pass^5")],
    layout: baseLayout(t, narrow, {
      xaxis: axis(t, `heuristics retrieved per turn (BM25, of ${retrieval.bank_size})`, {
        tickvals: x, ticktext: pts.map((p) => (p.k === 0 ? "0" : String(p.k))), range: [-0.3, x[x.length - 1] + 0.3],
      }, narrow),
      yaxis: axis(t, "success", { range: [0, 70], dtick: 10, ticksuffix: "%" }, narrow),
      annotations: [
        { x: x[x.length - 1], y: ref.msr, xanchor: "right", yanchor: "bottom", showarrow: false, text: "whole memory at start", font: { size: 11, color: t.tealInk } },
        { x: x[0], y: 44.3, xanchor: "left", yanchor: "top", showarrow: false, yshift: -4, text: "no memory", font: { size: 11, color: t.muted } },
      ],
    }),
  };
}

const retEl = byId("retrieval-chart");
if (retEl) void scrub(retEl, { draw: retrievalChart, effect: drawLinesAt });
