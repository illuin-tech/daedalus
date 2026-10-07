// [theme: auto] → light → dark → auto. "auto" follows the OS; the choice is remembered
// per browser. Every change dispatches `themechange` so the chart can repaint.
const root = document.documentElement;
const button = document.querySelector<HTMLButtonElement>(".theme-toggle");
const label = button?.querySelector(".value");
const CYCLE = ["auto", "light", "dark"] as const;
type Theme = (typeof CYCLE)[number];

const current = (): Theme => (root.dataset.theme as Theme | undefined) ?? "auto";

function apply(theme: Theme) {
  if (theme === "auto") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    if (theme === "auto") localStorage.removeItem("theme");
    else localStorage.setItem("theme", theme);
  } catch {}
  if (label) label.textContent = theme;
  document.dispatchEvent(new Event("themechange"));
}

if (label) label.textContent = current();
button?.addEventListener("click", () => apply(CYCLE[(CYCLE.indexOf(current()) + 1) % CYCLE.length]));

// In "auto", an OS switch is a theme change too.
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  if (current() === "auto") document.dispatchEvent(new Event("themechange"));
});
