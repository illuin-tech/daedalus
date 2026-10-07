// Scroll-driven figures: a figure's progress p follows the scroll, both ways. p runs linearly
// from 0, when the top of the figure's box enters the window from below (or at the top of the
// page, if it is already in), to 1, when the whole box is in view, its bottom at the bottom of
// the window. With reduced motion, p is always 1.

export const REDUCED = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export const clamp01 = (x: number) => Math.min(1, Math.max(0, x));

export function progressOf(box: Element) {
  if (REDUCED()) return 1;
  const r = box.getBoundingClientRect();
  const start = Math.max(0, r.top + window.scrollY - window.innerHeight);
  const end = r.bottom + window.scrollY - window.innerHeight;
  return end <= start ? 1 : clamp01((window.scrollY - start) / (end - start));
}

/** Calls `update` now and on every scroll or resize, at most once per frame. */
export function onScroll(update: () => void) {
  update();
  if (REDUCED()) return;
  let queued = false;
  const tick = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; update(); });
  };
  window.addEventListener("scroll", tick, { passive: true });
  window.addEventListener("resize", tick);
}
