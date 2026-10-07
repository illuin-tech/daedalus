// Marks the section being read in the contents: the last section whose top has scrolled
// past the upper third of the viewport.
const sections = [...document.querySelectorAll<HTMLElement>("main section[id]")];
const items = [...document.querySelectorAll<HTMLElement>(".toc li[data-target]")];

function update() {
  const line = window.innerHeight / 3;
  let active = sections[0]?.id;
  for (const s of sections) if (s.getBoundingClientRect().top <= line) active = s.id;
  // At the very bottom a short last section never reaches the line: it is the one read.
  const bottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2;
  if (bottom && sections.length) active = sections[sections.length - 1].id;
  for (const li of items) li.classList.toggle("active", li.dataset.target === active);
}

let queued = false;
window.addEventListener("scroll", () => {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; update(); });
}, { passive: true });
window.addEventListener("resize", update);
update();

// A wide figure runs over the sidebar's column: fade the sidebar while one is in view.
const visible = new Set<Element>();
const watcher = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.intersectionRatio >= 0.2) visible.add(e.target);
    else visible.delete(e.target);
  }
  document.body.classList.toggle("sidebar-faded", visible.size > 0);
}, { threshold: [0, 0.2, 0.5] });
document.querySelectorAll(".wide").forEach((el) => watcher.observe(el));

// Jumping to a section inside a closed <details> (the appendix) opens it first.
function openTarget() {
  const target = location.hash && document.getElementById(location.hash.slice(1));
  const box = target && (target.querySelector("details") ?? target.closest("details"));
  if (box && !box.open) box.open = true;
}
window.addEventListener("hashchange", openTarget);
openTarget();
