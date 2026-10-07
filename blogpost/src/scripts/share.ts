// [share this blogpost]: the system share sheet on phones and tablets, where it is the natural
// gesture; elsewhere the link (without any ?query or #section) is copied to the clipboard.
// Also the [copy] buttons, below.
const button = document.querySelector<HTMLButtonElement>(".share");
const label = button?.querySelector<HTMLElement>(".value");

if (button && label) {
  const idle = label.textContent ?? "";
  let timer: number | undefined;
  const say = (text: string) => {
    label.textContent = text;
    window.clearTimeout(timer);
    timer = window.setTimeout(() => (label.textContent = idle), 2200);
  };

  button.addEventListener("click", async () => {
    const url = location.origin + location.pathname;
    const touch = window.matchMedia("(pointer: coarse)").matches;
    if (touch && navigator.share) {
      try { await navigator.share({ title: document.title, url }); } catch { /* dismissed */ }
      return;
    }
    try {
      await navigator.clipboard.writeText(url);
      say("link copied ✓");
    } catch {
      say(url);  // no clipboard access: show the link to copy by hand
    }
  });
}

// [copy] buttons (the citation): copy the text of the element named in `data-copy`.
for (const b of document.querySelectorAll<HTMLButtonElement>("[data-copy]")) {
  const source = document.querySelector<HTMLElement>(b.dataset.copy!);
  if (!source) continue;
  b.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(source.innerText);
      b.textContent = "[copied ✓]";
    } catch {
      // no clipboard access: select the text, to copy by hand
      getSelection()?.selectAllChildren(source);
      b.textContent = "[selected]";
    }
    window.setTimeout(() => (b.textContent = "[copy]"), 2000);
  });
}

export {};
