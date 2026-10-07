// The walkthrough of fig. 3: one real generation session of the 90-session AppWorld run
// (session 46, outputs/generation/appworld_90sessions/sessions/session_46.json), framed by the
// one-time survey before it and the consolidation after the last session. Texts are condensed
// from the session file and its Solver traces (traces/229360a_2_*);
// `lit` / `flow` name parts of the diagram (scripts/pipeline_diagram.py).

export interface Step {
  actor: string;
  summary: string;
  /** Trusted, authored HTML. */
  detail: string;
  lit: string[];
  flow: string[];
}

const task = (label: string, text: string) => `<span class="lbl">${label}</span><q>${text}</q>`;
// A heuristic as a diff against the previous one: + added, - removed, the rest unchanged.
const OP = { "+": "add", "-": "del", " ": "ctx" } as const;
const diff = (lines: [keyof typeof OP, string][]) =>
  `<span class="diff">${lines.map(([op, t]) => `<span class="${OP[op]}"><i>${op === " " ? "&nbsp;" : op}</i><span>${t}</span></span>`).join("")}</span>`;

const SOLVER_JUDGE = { lit: ["loop", "solver", "judge"], flow: ["a-trace"] };
const EXTRACTOR = { lit: ["loop", "extractor"], flow: ["a-fail", "a-heur"] };

export const STEPS: Step[] = [
  {
    actor: "surveyor",
    summary: "maps the environment once",
    detail:
      `Before the first session, the Surveyor explores AppWorld and sets a coverage goal: a share
       of tasks per area, <span class="k">Spotify 22%</span>, <span class="k">Todoist 18%</span>,
       <span class="k">Splitwise 17%</span>, <span class="k">Venmo 16%</span>,
       <span class="k">Phone 12%</span>… Every session sees the goal and a tally of the tasks
       accepted so far.`,
    lit: ["surveyor"],
    flow: ["a-tags"],
  },
  {
    actor: "explorer",
    summary: "proposes a task, T1",
    detail:
      `Session 46 of 90. Guided by the coverage goal and its guideline memory, the Explorer spends
       19 turns in the apps, then proposes a task, solves it itself to prove it feasible, and
       writes the conditions a correct answer must meet.
       ${task("task T1", "Please clean up my Todoist by marking done any subtask that's still open even though its parent task is already completed.")}
       ${task("success conditions", "Every Todoist subtask that started unfinished under a parent task that was already completed is now completed. No completed Todoist task still has an unfinished subtask.")}`,
    lit: ["explorer", "guidelines"],
    flow: ["a-guide", "a-task", "a-cond"],
  },
  {
    actor: "solver+judge",
    summary: "T1 is too easy",
    detail:
      `The Solver attempts T1 from a fresh state each time, and the Judge checks every trajectory
       against the conditions: <span class="res ok">✓✓✓</span>
       The Solver succeeds three times in a row, in 10, 14 and 10 turns, and never fails: T1 is
       <strong>too easy</strong>, and nothing from it is kept.`,
    ...SOLVER_JUDGE,
  },
  {
    actor: "explorer",
    summary: "refinement #1, T2",
    detail:
      `The verdict goes back to the Explorer, which makes the task harder by adding the opposite
       mismatch, and checks that it is still solvable.
       ${task("task T2", "…so task and subtask completion statuses line up: mark done any open subtask whose parent task is already completed, and also mark done any open parent task whose subtasks are all already finished. Tell me how many items you fixed.")}
       ${task("success conditions", "Every open subtask under a completed parent, and every open parent whose subtasks were all completed, is now completed. The user is told how many items changed.")}`,
    lit: ["explorer"],
    flow: ["a-feedback", "a-task", "a-cond"],
  },
  {
    actor: "solver+judge",
    summary: "T2 is too easy",
    detail:
      `The Solver tries the new task T2, from a fresh state and with no heuristic yet.
       <span class="res ok">✓✓✓</span> It succeeds three times in a row again, in 9, 13 and 8
       turns: a mirror case and a count do not make the sweep harder, and T2 is still
       <strong>too easy</strong>.`,
    ...SOLVER_JUDGE,
  },
  {
    actor: "explorer",
    summary: "refinement #2, T3",
    detail:
      `The verdict goes back to the Explorer once more. Rather than add a case, it changes one
       rule: a completed task with an unfinished subtask must now be reopened, so the two kinds
       of mismatch need opposite fixes.
       ${task("task T3", "…so parent tasks and subtasks agree on what's finished: if a task still has an unfinished subtask it shouldn't be completed, and if all its subtasks are already done the task should be completed too. Tell me how many tasks you fixed.")}
       ${task("success conditions", "Every task that started out completed with an unfinished subtask is now open, and every task that started out open with all its subtasks done is now completed. The user is told how many parent tasks changed.")}`,
    lit: ["explorer"],
    flow: ["a-feedback", "a-task", "a-cond"],
  },
  {
    actor: "solver+judge",
    summary: "T3 fails",
    detail:
      `The Solver tries T3, from a fresh state and with no heuristic yet, in 9 turns.
       <span class="res"><b>✗</b></span> The Judge rejects the answer: the Solver checked the
       parent tasks of a single project, so the mismatches elsewhere in Todoist were never fixed.`,
    ...SOLVER_JUDGE,
  },
  {
    actor: "extractor",
    summary: "writes h1",
    detail:
      `The Extractor reads the failed trajectory, without seeing the success conditions, and
       writes a heuristic, h1, which the Solver gets in context on its next attempt:
       ${diff([
         ["+", "to reconcile parents with subtasks, inspect <em>every</em> parent with <code>num_sub_tasks &gt; 0</code>, reading its subtasks with <code>todoist.show_sub_tasks</code> before updating anything"],
         ["+", "a completed parent with an unfinished subtask: reopen it with <code>todoist.update_task(…, is_completed=False)</code>"],
         ["+", "before completing an open parent, check that all of its subtasks are done"],
         ["+", "re-read the changed parents and their subtasks, then submit the count with <code>complete_task</code>"],
       ])}`,
    ...EXTRACTOR,
  },
  {
    actor: "solver+judge",
    summary: "three successes in a row",
    detail:
      `<span class="res ok">✓✓✓</span> With h1 in context, the Solver succeeds three times in a
       row, in 8, 23 and 14 turns. A single success could be luck; three consecutive ones show
       that h1 fixes the failure.`,
    ...SOLVER_JUDGE,
  },
  {
    actor: "memory",
    summary: "h1 is accepted",
    detail:
      `h1 turned a failure into repeated success, so it enters the heuristic memory. Heuristics
       from tasks that stay too easy or too hard never enter it: this is the only way in.`,
    lit: ["hbank"],
    flow: ["a-bank"],
  },
  {
    actor: "guidelines",
    summary: "a lesson for the Explorer",
    detail:
      `The session needed refinements, so its history is distilled into the Explorer's
       guidelines, which are rewritten. Among them, for every later session:<q>if a cleanup sweep
       is still too easy, widen it to one natural consistency invariant across the whole slice,
       especially parent/child or group/member agreement […]; don't rely on bolted-on mirror cases
       or extra reporting to raise difficulty</q>`,
    lit: ["guidelines"],
    flow: ["a-lesson"],
  },
  {
    actor: "consolidator",
    summary: "one memory for test time",
    detail:
      `After the last of the 90 sessions, 81 heuristics have been accepted. The Consolidator merges
       them in one call into 65 non-redundant ones, and that memory is injected, whole, at the
       start of every test task.`,
    lit: ["hbank", "consolidator", "memory"],
    flow: ["a-consolidate", "a-deploy"],
  },
];
