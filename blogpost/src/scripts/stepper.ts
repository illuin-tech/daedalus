// Step-by-step figures (figs. 2, 3 and 4, the ablation). A `[data-stepper]` element carries its steps as
// JSON ({lit, flow} per step, written back as `data-lit` / `data-flow` for the diagram), and
// marks its parts by index: `[data-row]` (done / now), `[data-detail]` (only the current one is
// shown), `[data-go]` buttons, `[data-act]` prev / next / play, and a `[data-count]`.
// Play advances one step every 5 s and stops on the last. With `data-wrap`, "next" on the last
// step goes back to the first (the element gets `at-end` there). With `data-scrub`, the steps
// follow the scroll both ways: each row fades in (its `--q`, 0 to 1) as its top travels from
// the bottom of the window to 75% of its height, and the current step is the last row fully in.
// `?step=N` opens every other stepper at N.
import { clamp01, onScroll, REDUCED } from "./scroll";

const STEP_MS = 5000;

for (const fig of document.querySelectorAll<HTMLElement>("[data-stepper]")) {
  const steps: { lit: string[]; flow: string[] }[] = JSON.parse(fig.dataset.steps ?? "[]");
  const rows = [...fig.querySelectorAll<HTMLElement>("[data-row]")];
  const details = [...fig.querySelectorAll<HTMLElement>("[data-detail]")];
  const bar = [...fig.querySelectorAll<HTMLElement>(".pl-bar button")];
  const count = fig.querySelector<HTMLElement>("[data-count]");
  const play = fig.querySelector<HTMLButtonElement>("[data-act=play]");
  const wrap = fig.dataset.wrap !== undefined;
  let current = 0;
  let timer: number | undefined;

  function show(i: number) {
    current = Math.max(0, Math.min(steps.length - 1, i));
    const { lit, flow } = steps[current];
    fig.dataset.lit = lit.join(" ");   // the diagram's colours follow these (DiagramStyles.astro)
    fig.dataset.flow = flow.join(" ");
    fig.dataset.step = String(current);
    for (const el of rows) {
      const k = Number(el.dataset.row);
      el.classList.toggle("done", k < current);
      el.classList.toggle("now", k === current);
    }
    details.forEach((d) => d.classList.toggle("on", Number(d.dataset.detail) === current));
    if (details[0]?.parentElement) details[0].parentElement.scrollTop = 0;  // phones scroll the detail (fig. 3)
    bar.forEach((b, k) => {
      b.classList.toggle("done", k < current);
      b.classList.toggle("now", k === current);
    });
    fig.classList.toggle("at-end", current === steps.length - 1);
    if (count) count.textContent = String(current + 1).padStart(2, "0");
  }

  function setPlaying(on: boolean) {
    window.clearInterval(timer);
    timer = undefined;
    fig.classList.toggle("playing", on);
    if (on) {
      if (current === steps.length - 1) show(0); // replay from the start
      timer = window.setInterval(() => {
        if (current >= steps.length - 1) return setPlaying(false);
        show(current + 1);
        if (current === steps.length - 1) setPlaying(false);
      }, STEP_MS);
    }
    if (play) {
      const atEnd = !on && current === steps.length - 1;
      play.textContent = on ? "‖ pause" : atEnd ? "▶ replay" : "▶ play";
    }
  }

  // A manual move keeps playback going, from the new step, with a fresh 5 s.
  const go = (i: number) => {
    show(i);
    if (timer !== undefined) setPlaying(current < steps.length - 1);
    else setPlaying(false);
  };

  fig.addEventListener("click", (e) => {
    const t = (e.target as HTMLElement).closest<HTMLElement>("button");
    if (!t) return;
    if (t.dataset.go !== undefined) go(Number(t.dataset.go));
    else if (t.dataset.act === "prev") go(current - 1);
    else if (t.dataset.act === "next") go(wrap && current === steps.length - 1 ? 0 : current + 1);
    else if (t.dataset.act === "play") setPlaying(timer === undefined);
  });

  fig.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") { e.preventDefault(); go(current + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(current - 1); }
  });

  const start = Number(new URLSearchParams(location.search).get("step")) || 1;
  show(Math.min(start, steps.length) - 1);

  // Scroll-driven: each row's own position sets its fade, from 0 when its top enters at the
  // bottom of the window to 1 when it is a quarter of the window's height above it.
  if (fig.dataset.scrub !== undefined) {
    onScroll(() => {
      const vh = window.innerHeight;
      let k = 0;
      for (const r of rows) {
        const q = REDUCED() ? 1 : clamp01((vh - r.getBoundingClientRect().top) / (0.25 * vh));
        r.style.setProperty("--q", String(q));
        if (q > 0.99) k = Math.max(k, Number(r.dataset.row));
      }
      if (k !== current) show(k);
    });
  }
}
