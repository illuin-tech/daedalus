// The generated tasks as a test set: the scatter of the paper's Figure 8. Each model appears
// twice (MSR filled, pass³ hollow); the points rise into place, lowest first, as the page
// scrolls (and sink back when scrolling up).
import Plotly from "plotly.js-basic-dist-min";
import figure from "../data/correlation.json";
import { SYMBOLS } from "../data/models";
import { axis, baseLayout, hexA, risePointsAt, scrub, type Tokens } from "./chart-theme";

const chart = document.getElementById("correlation-chart");
const METRICS = ["msr", "pass3"] as const;
const LABEL = { msr: "MSR", pass3: "pass³" } as const;

function scatter(t: Tokens, narrow: boolean) {
  const data = METRICS.flatMap((metric) =>
    figure.models.map((model) => {
      const msr = metric === "msr";
      return {
        type: "scatter", mode: "markers",
        x: [model[metric].real * 100], y: [model[metric].generated * 100],
        name: model.id,
        marker: {
          symbol: SYMBOLS[model.id], size: 15,
          color: msr ? t.teal : hexA(t.orange, 0.28),
          line: { color: msr ? t.ink : t.orange, width: 2 },
        },
        hovertemplate:
          `<b>${model.id}</b> · ${LABEL[metric]}<br>` +
          "AppWorld  %{x:.1f}%<br>generated %{y:.1f}%<extra></extra>",
      };
    }),
  );
  return {
    data,
    layout: baseLayout(t, narrow, {
      xaxis: axis(t, "AppWorld test_normal", { range: [-5, 71], dtick: narrow ? 20 : 10, ticksuffix: "%" }, narrow),
      yaxis: axis(t, "DAEDALUS-generated tasks", { range: [-5, 107], dtick: narrow ? 20 : 10, ticksuffix: "%" }, narrow),
    }),
  };
}

if (chart) {
  const fig = chart.closest("figure")!;
  const ids = METRICS.flatMap(() => figure.models.map((m) => m.id));
  void scrub(chart, { draw: scatter, effect: risePointsAt });

  // Legend: hovering (or tapping) a model brings its two points forward.
  const focus = (id: string | null) => {
    const opacity = ids.map((m) => (id === null || m === id ? 1 : 0.15));
    void Plotly.restyle(chart, { opacity }, [...ids.keys()]);
    fig.querySelectorAll<HTMLElement>(".legend li").forEach((li) => li.classList.toggle("on", li.dataset.model === id));
  };
  let pinned: string | null = null;
  fig.querySelectorAll<HTMLElement>(".legend li").forEach((li) => {
    const id = li.dataset.model ?? null;
    li.addEventListener("mouseenter", () => focus(id));
    li.addEventListener("mouseleave", () => focus(pinned));
    li.addEventListener("click", () => { pinned = pinned === id ? null : id; focus(pinned); });
  });
}
