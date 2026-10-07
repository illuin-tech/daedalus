// Tables: benchmark tabs (table 1, on narrow screens), a column highlight that follows the
// pointer, and a reveal for figures that animate once in view.
for (const fig of document.querySelectorAll<HTMLElement>("[data-bench-tabs]")) {
  const table = fig.querySelector<HTMLElement>("table")!;
  fig.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach((b) =>
    b.addEventListener("click", () => {
      table.dataset.show = b.dataset.tab;
      fig.querySelectorAll("[data-tab]").forEach((x) => x.classList.toggle("on", x === b));
    }));
}

for (const table of document.querySelectorAll<HTMLTableElement>("table.t")) {
  table.addEventListener("mouseover", (e) => {
    const cell = (e.target as HTMLElement).closest("td, th") as HTMLTableCellElement | null;
    if (!cell || !cell.parentElement?.closest("tbody")) return;
    const col = cell.cellIndex;
    table.querySelectorAll("tbody td").forEach((td) => td.classList.toggle("col-hl", (td as HTMLTableCellElement).cellIndex === col && col > 0));
  });
  table.addEventListener("mouseleave", () => table.querySelectorAll(".col-hl").forEach((td) => td.classList.remove("col-hl")));
}

const reveal = new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); reveal.unobserve(e.target); }
}, { threshold: 0.3 });
document.querySelectorAll("[data-reveal]").forEach((el) => reveal.observe(el));

// Metric toggles (table 4): the figure's `data-metric` picks which values the bars show.
for (const seg of document.querySelectorAll<HTMLElement>("[data-metric-toggle]")) {
  const fig = seg.closest<HTMLElement>("figure")!;
  seg.querySelectorAll<HTMLButtonElement>("[data-metric]").forEach((b) =>
    b.addEventListener("click", () => {
      fig.dataset.metric = b.dataset.metric;
      seg.querySelectorAll("[data-metric]").forEach((x) => x.classList.toggle("on", x === b));
    }));
}
