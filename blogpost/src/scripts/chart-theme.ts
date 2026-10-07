// Shared look for every Plotly chart: colours come from the page's tokens (so light and dark
// both work), one mono font, recessive grid and frame. `mount` draws a chart, redraws it on a
// theme change or when it crosses the narrow breakpoint, and can play an entrance once it
// scrolls into view.
import Plotly from "plotly.js-basic-dist-min";
import { clamp01, onScroll, progressOf } from "./scroll";

export type Tokens = ReturnType<typeof tokens>;

export function tokens() {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return {
    bg: v("--bg"), ink: v("--ink"), muted: v("--muted"), rule: v("--rule"),
    ruleStrong: v("--rule-strong"), teal: v("--teal"), orange: v("--orange"),
    tealInk: v("--teal-ink"), orangeInk: v("--orange-ink"), font: v("--mono"),
    roles: {
      explorer: v("--role-explorer"), solver: v("--role-solver"),
      judge: v("--role-judge"), extraction: v("--role-extractor"),
    } as Record<string, string>,
  };
}

export function hexA(hex: string, alpha: number) {
  const n = parseInt(hex.replace("#", ""), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

/** The colour `p` of the way from `a` to `b` (both #rrggbb). */
export function mix(a: string, b: string, p: number) {
  const rgb = (h: string) => { const n = parseInt(h.replace("#", ""), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
  const [x, y] = [rgb(a), rgb(b)];
  return `rgb(${x.map((c, i) => Math.round(c + (y[i] - c) * p)).join(",")})`;
}

export const NARROW = 520;

export function axis(t: Tokens, title: string, extra: Record<string, unknown> = {}, narrow = false) {
  return {
    title: { text: title, standoff: 10, font: { size: narrow ? 11 : 12.5, color: t.muted } },
    tickfont: { size: narrow ? 10 : 11.5, color: t.muted },
    gridcolor: t.rule, griddash: "dot", zeroline: false,
    showline: true, mirror: true, linecolor: t.ruleStrong, linewidth: 1, ticks: "",
    ...extra,
  };
}

export function baseLayout(t: Tokens, narrow: boolean, extra: Record<string, unknown> = {}) {
  return {
    paper_bgcolor: t.bg, plot_bgcolor: t.bg,
    font: { family: t.font, color: t.ink },
    margin: narrow ? { l: 46, r: 8, t: 8, b: 46 } : { l: 58, r: 14, t: 10, b: 52 },
    hovermode: "closest",
    hoverlabel: { bgcolor: t.ink, bordercolor: t.ink, font: { family: t.font, size: 12, color: t.bg } },
    showlegend: false,
    ...extra,
  };
}

export const CONFIG = { responsive: true, displayModeBar: false, scrollZoom: false, doubleClick: false };

type Draw = (t: Tokens, narrow: boolean) => { data: unknown[]; layout: Record<string, unknown> };
/** How a chart enters once it scrolls into view: from another frame, lines drawn on, or
 * points rising into place from the lowest to the highest. */
type Enter = Draw | "draw-lines" | "rise-points";

const REDUCE = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Line charts: each solid line is hidden behind its own dash, then drawn from left to right;
// markers and error bars fade in after it. Dotted reference lines are left alone.
function hideLines(el: HTMLElement) {
  el.querySelectorAll<SVGPathElement>(".scatterlayer .js-line").forEach((path) => {
    if (path.style.strokeDasharray && path.style.strokeDasharray !== "none") return;
    const len = path.getTotalLength();
    path.style.transition = "none";
    path.style.strokeDasharray = `${len} ${len}`;
    path.style.strokeDashoffset = String(len);
  });
  el.querySelectorAll<SVGElement>(".scatterlayer .points, .scatterlayer .errorbars").forEach((g) => {
    g.style.transition = "none";
    g.style.opacity = "0";
  });
}

function drawLines(el: HTMLElement) {
  void el.getBoundingClientRect(); // commit the hidden state before transitioning
  el.querySelectorAll<SVGPathElement>(".scatterlayer .js-line").forEach((path) => {
    if (!path.style.strokeDashoffset) return;
    path.style.transition = "stroke-dashoffset 1.4s cubic-bezier(.45,0,.25,1)";
    path.style.strokeDashoffset = "0";
  });
  el.querySelectorAll<SVGElement>(".scatterlayer .points, .scatterlayer .errorbars").forEach((g) => {
    g.style.transition = "opacity 0.5s ease 1.1s";
    g.style.opacity = "1";
  });
}

// Scatter charts: every trace (one point each) starts lower and transparent, and rises into
// place in order of height, lowest first. The inline styles are cleared once it is done, so
// hover and restyle work as usual.
const RISE_PX = 36, RISE_MS = 700, RISE_STAGGER = 55;
const traces = (el: HTMLElement) => [...el.querySelectorAll<SVGGElement>(".scatterlayer .trace")];

function hidePoints(el: HTMLElement) {
  for (const g of traces(el)) {
    g.style.transition = "none";
    g.style.transform = `translateY(${RISE_PX}px)`;
    g.style.opacity = "0";
  }
}

function risePoints(el: HTMLElement) {
  void el.getBoundingClientRect();
  const byHeight = traces(el)
    .map((g) => ({ g, y: g.querySelector(".point")?.getBoundingClientRect().top ?? 0 }))
    .sort((a, b) => b.y - a.y);  // lowest on screen first
  byHeight.forEach(({ g }, i) => {
    g.style.transition = `transform ${RISE_MS}ms cubic-bezier(.2,.7,.3,1) ${i * RISE_STAGGER}ms, opacity ${RISE_MS}ms ease ${i * RISE_STAGGER}ms`;
    g.style.transform = "none";
    g.style.opacity = "1";
  });
  window.setTimeout(() => traces(el).forEach((g) => {
    g.style.transition = g.style.transform = "";
    if (g.style.opacity === "1") g.style.opacity = "";
  }), RISE_MS + byHeight.length * RISE_STAGGER + 50);
}

/**
 * Draws `el` with `draw`, keeps it in sync with the theme and the breakpoint, and plays an
 * entrance once the chart is in view: `enter` is either the frame to animate from,
 * "draw-lines" to draw the lines on, or "rise-points" to raise the points into place.
 * Resolves once the chart is first drawn.
 */
export function mount(el: HTMLElement, draw: Draw, enter?: Enter): Promise<void> {
  const narrow = () => el.clientWidth < NARROW;
  const fromFrame = typeof enter === "function" ? enter : undefined;
  let entered = !enter || REDUCE();
  const render = () => {
    const t = tokens();
    const { data, layout } = (entered || !fromFrame ? draw : fromFrame)(t, narrow());
    return Plotly.newPlot(el, data as never, layout, CONFIG).then(() => {
      if (entered) return;
      if (enter === "draw-lines") hideLines(el);
      else if (enter === "rise-points") hidePoints(el);
    });
  };
  return render().then(() => {
    let wasNarrow = narrow();
    new ResizeObserver(() => {
      if (narrow() !== wasNarrow) { wasNarrow = narrow(); void render(); }
      else void Plotly.Plots.resize(el);  // e.g. the chart was drawn inside a closed <details>
    }).observe(el);
    document.addEventListener("themechange", () => void render());
    if (entered) return;
    const io = new IntersectionObserver((entries) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      io.disconnect();
      entered = true;
      if (enter === "draw-lines") return drawLines(el);
      if (enter === "rise-points") return risePoints(el);
      const { data, layout } = draw(tokens(), narrow());
      void Plotly.animate(el, { data, layout }, { transition: { duration: 1100, easing: "cubic-in-out" }, frame: { duration: 1100, redraw: false } });
    }, { threshold: 0.45 });
    io.observe(el);
  });
}

/**
 * A chart whose animation follows the scroll, both ways (progress p, see scroll.ts). Either
 * `frame(p)` redraws the chart at p (0 = start, 1 = end), or the chart is drawn once with `draw`
 * and `effect(el, p)` moves its SVG; both can be given.
 */
export function scrub(
  el: HTMLElement,
  { frame, draw, effect }: { frame?: (p: number) => Draw; draw?: Draw; effect?: (el: HTMLElement, p: number) => void },
): Promise<void> {
  const narrow = () => el.clientWidth < NARROW;
  const box = el.closest("figure") ?? el;
  let t = tokens();
  let p = progressOf(box);
  const render = (fresh: boolean) => {
    const { data, layout } = (frame ? frame(p) : draw!)(t, narrow());
    return (fresh ? Plotly.newPlot : Plotly.react)(el, data as never, layout, CONFIG).then(() => effect?.(el, p));
  };
  return render(true).then(() => {
    let wasNarrow = narrow();
    new ResizeObserver(() => {
      if (narrow() !== wasNarrow) { wasNarrow = narrow(); void render(true); }
      else void Promise.resolve(Plotly.Plots.resize(el)).then(() => effect?.(el, p));
    }).observe(el);
    document.addEventListener("themechange", () => { t = tokens(); void render(true); });
    onScroll(() => {
      const q = progressOf(box);
      if (q === p) return;
      p = q;
      if (frame) void render(false);
      else effect?.(el, p);
    });
  });
}

/** Scrubbed line chart: each solid line is drawn from left to right over the first 80% of the
 * scroll, its markers and error bars fading in over the last 25%. Dotted lines stay put. */
export function drawLinesAt(el: HTMLElement, p: number) {
  el.querySelectorAll<SVGPathElement>(".scatterlayer .js-line").forEach((path) => {
    if (!path.dataset.len) {
      if (path.style.strokeDasharray && path.style.strokeDasharray !== "none") return;  // dotted
      path.dataset.len = String(path.getTotalLength());
    }
    const len = Number(path.dataset.len);
    path.style.transition = "none";
    path.style.strokeDasharray = `${len} ${len}`;
    path.style.strokeDashoffset = String(len * (1 - clamp01(p / 0.8)));
  });
  el.querySelectorAll<SVGElement>(".scatterlayer .points, .scatterlayer .errorbars").forEach((g) => {
    g.style.transition = "none";
    g.style.opacity = String(clamp01((p - 0.75) / 0.25));
  });
}

/** Scrubbed scatter: each trace (one point) rises into place and fades in, lowest first, each
 * over 40% of the scroll, staggered. At p = 1 the inline styles are cleared, so restyling
 * (e.g. the legend's focus) works as usual. */
export function risePointsAt(el: HTMLElement, p: number) {
  const gs = traces(el);
  const data = (el as unknown as { data: { y: number[] }[] }).data;
  const order = gs.map((_, i) => i).sort((a, b) => (data[a]?.y[0] ?? 0) - (data[b]?.y[0] ?? 0));
  const span = 0.4, step = gs.length > 1 ? (1 - span) / (gs.length - 1) : 0;
  order.forEach((i, rank) => {
    const g = gs[i];
    g.style.transition = "none";
    if (p >= 1) { g.style.transform = g.style.opacity = ""; return; }
    const q = clamp01((p - rank * step) / span);
    g.style.transform = `translateY(${RISE_PX * (1 - q)}px)`;
    g.style.opacity = String(q);
  });
}
