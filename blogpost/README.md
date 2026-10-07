# DAEDALUS blog post

A single-page [Astro](https://astro.build) site: an all-monospace, ASCII-styled write-up of the
paper, with the figures redrawn in [Plotly](https://plotly.com/javascript/).

## Run locally

Astro 7 needs Node.js 22.12 or later (`.nvmrc` pins 22.13).

```sh
nvm use
npm install
npm run dev        # http://localhost:4321/daedalus/, reloads on save
npm run build      # static site in dist/; `npm run preview` serves it
npm run check      # type-check the .astro and .ts files
```

## Deploy

`.github/workflows/blog.yml` builds the site and publishes it on GitHub Pages at
https://illuin-tech.github.io/daedalus/, on every push to `main` that touches `blogpost/` (or
by hand, from the Actions tab). It needs Settings → Pages → Source set to "GitHub Actions". The
site lives under `/daedalus/` (`base` in `astro.config.mjs`): link a file of `public/` through
`import.meta.env.BASE_URL` (the `asset()` helper), never as a bare `/path`; in CSS, use files
under `src/` with relative URLs (as the fonts do).

## Layout

| path | what |
| --- | --- |
| `src/pages/index.astro` | the post: title block, sections, figure markup |
| `src/layouts/Post.astro` | page shell: sidebar contents (dimmed until hovered; collapsed into `[+] contents` below 1180px), `[share this blogpost]` and theme switch, footer |
| `src/components/Maze.astro` | the header labyrinth that writes DAEDALUS (its exit arrow is a button, clickable only while the maze is hovered, that opens `Myth.astro`; to its right, above 1000px, a clickable "why this name?" slowly blinks between orange at 50% opacity and transparent). The orange route draws itself, stays 2.5 s, fades out and draws itself again (a small script; once only with reduced motion): each letter (a 3×5 pixel font) is a room outlined in teal with a soft teal fill, the rest a maze generated at build time. Of 2000 seeds it keeps the maze whose route (orange) weaves between the letters most often. Phones get a two-line DAED / ALUS version |
| `src/components/Regimes.astro` | fig. 2 (end of section 01): three ways to meet a new environment, one row each, flipping down into place as the page scrolls, hinged at its top like a split-flap board, and back up when scrolling up (no controls): no memory (the same error on every test task), memory methods (learn from curated training tasks and a verifier), DAEDALUS (no training tasks, no human priors). Same tinted-cell look as fig. 4 |
| `src/components/Myth.astro` | the note on the name: a modal `<dialog>` mapping the myth to the pipeline (Daedalus → Explorer, Theseus → Solver, Minos → Judge, Ariadne → Extractor, Icarus → refinement, Talos → Surveyor, the clew → Consolidator), opened by the exit arrow of the header maze; the page does not scroll behind it. Each character has a 64 px square picture: drop `public/myth/<slug>.png` (`daedalus`, `theseus`, `minotaur`, `minos`, `ariadne`, `thread`, `icarus`, `talos`, `clew`) and the next build shows it in place of the placeholder; a picture's source (`credit` in `CAST`) is a tiny "source" link under it |
| `src/components/Pipeline.astro`, `src/scripts/pipeline.ts` | fig. 3: the generation pipeline (paper Figures 2 and 3) as ASCII art beside a log of one real session; prev / play (one step every 5 s) / next, the arrow keys, and `?step=N` to open at a step. Each component keeps its colour from the paper's figures (outline colours in `COLOR`): full outline and faint fill when it takes part in the step, fainter otherwise |
| `.wide` (in `global.css`) | a figure at 75% of the window width; the sidebar fades further while one is on screen (`src/scripts/toc.ts`) |
| `src/data/pipeline-steps.ts` | the twelve steps (the Solver and Judge in one step, the Extractor in its own; heuristics shown as + / − diffs): text condensed from AppWorld session 46 (`outputs/daedalus/appworld/sessions/session_46.json`: a task twice too easy, refined into one that fails, fixed by a single heuristic), and the diagram parts each one lights up. On phones the step's text scrolls in a fixed-height area |
| `scripts/pipeline_diagram.py` | draws the diagram and writes `src/data/pipeline-diagram.json`; rerun it after editing the drawing (`python3 scripts/pipeline_diagram.py`) |
| `src/components/MemoryExample.astro` | fig. 4: one AppWorld test task, turn by turn (each step comes in as the page scrolls, the run without memory sliding in from the left and the run with memory from the right, and goes back out when scrolling up; no controls), in two tinted columns: the run without memory goes off track and answers wrong, the run with memory follows two heuristics and answers right. Condensed from run 2 of each; data in `src/data/memory-example.json`, from the traces it names |
| `src/components/ResultsTable.astro`, `SmallTables.astro` | table 1 (one tab per benchmark; each baseline's name links to its arXiv page, `PAPERS` in `src/data/tables.ts`) and appendix tables A1–A2 (`src/data/tables.ts`, transcribed from the preprint) |
| `src/components/Injection.astro` | table 4 as a bar list: one bar per injection policy on a shared scale, MSR / pass^5 toggle, the no-memory baseline dotted, cost per run on the right |
| `src/components/Takeaway.astro` | a takeaway card (styled like the tl;dr), one or two self-contained sentences per section |
| `src/components/TransferMatrix.astro` | table 2 as a matrix: cells shaded by gain, filled in once in view, hover names both scores |
| `src/components/Ablation.astro` | table 3 as a stepper: the diagram (`DiagramArt.astro`) lights up each component as it is added, the table fills row by row |
| `src/components/DiagramArt.astro`, `DiagramStyles.astro` | the pipeline diagram and its colour rules (once per page), shared by fig. 3 and table 3 |
| `src/scripts/stepper.ts` | prev / play / next for fig. 3 and table 3 (`data-wrap`: "next" on the last step restarts; `?step=N` opens them at step N); with `data-scrub` (figs. 2 and 4), the steps follow the scroll instead, each row coming in as its top travels from the bottom of the window to 75% of its height |
| `src/scripts/scroll.ts` | the scroll progress of a figure, shared by the scroll-driven charts (figs. 1, 5, 6, 7): 0 when the top of its box enters the window (or at the top of the page), 1 when the whole box is in view, linear in between, both ways; always 1 with reduced motion |
| `src/pages/index.astro`: links | `ARXIV` (the preprint, also in the BibTeX), `CODE`, `DATASET` (🤗 `illuin/daedalus-traces`, also linked under the tl;dr) and `ILLUIN` feed the header buttons, the sidebar and the link boxes before the citation. Every author name is underlined; hovering or tapping it fades the name and shows its LinkedIn, X and email icons over it (optional `linkedin` / `x` URLs and `mail` address: an icon shows only when set) |
| `public/illuin-logo.svg` | the Illuin logo, next to the affiliation (written one colour per letter, `ILLUIN_COLORS`) and in the Illuin link box |
| `src/scripts/chart-theme.ts` | shared Plotly look (page tokens, theme changes) and the entrance once in view: from another frame, lines drawn on (`draw-lines`), or points rising lowest first (`rise-points`); or, with `scrub`, an animation that follows the scroll both ways (`scroll.ts`): a frame redrawn at each progress, or an effect on the drawn SVG (`drawLinesAt`: lines drawn on; `risePointsAt`: points rising lowest first) |
| `src/scripts/charts.ts` | figs. 1, 5, 6 and A1: model families (at the top of the page, before the tl;dr, each model moving from no memory to memory as the page scrolls, and back when scrolling up), generation cost by role (built with the scroll, one role's layer after the other from the bottom up, each bar's total riding on top and counting up; colours: the `--role-*` tokens), retrieval per turn (its lines drawn with the scroll), sessions (appendix); data in `src/data/fig-*.json`, exported with the paper's own plot scripts |
| `src/scripts/correlation.ts` | fig. 7: the scatter, points rising into place bottom to top as the page scrolls (and sinking back when scrolling up); the legend singles a model out |
| `src/scripts/tables.ts` | table tabs, column highlight, metric toggles (table 4), and reveal-on-scroll |
| `src/scripts/theme.ts`, `src/scripts/toc.ts` | `[theme: auto/light/dark]` switch, and the contents' current-section marker |
| `src/scripts/share.ts` | `[share this blogpost]`: the system share sheet on touch devices, otherwise copies the link; and the `[copy]` button of the BibTeX box (section @, `bibtex` in `index.astro`) |
| `src/styles/global.css` | color tokens (light and dark), the page grid, every component style |
| `src/data/` | the figure's data (`correlation.json`) and the model markers (`models.ts`) |
| `src/fonts/` | JetBrains Mono 2.304 (OFL), the full glyph set: box drawing, arrows, Greek; bundled by the build from `global.css` (relative URLs, so they follow the base path) |

Colors are tokens on `:root`: `--teal` and `--orange` are the figure's two series colors,
stepped per theme so the pair stays distinguishable (including for color-blind readers);
`--teal-ink` and `--orange-ink` are their text-safe variants (WCAG AA on the background).

## Figure data

`src/data/correlation.json` is a standalone snapshot of the data `plots/scripts/testset_correlation.py`
plots for the paper's Figure 8: nine models without memory, on the 81 tasks of the 90-session
AppWorld run (`outputs/evaluation-proxy/appworld/`) and on AppWorld
`test_normal` (`outputs/inference/appworld/`). The site reads only the snapshot. The τ values and
pair counts in the text are read from it, not typed in.
