"""Using generated tasks as a test set: freeze, run a model, judge, score.

`daedalus.scripts.tasks_as_test_set` is the entry point. The question this answers is whether
the tasks a generation run invented — instruction plus natural-language success conditions —
can rank models the way a curated benchmark does.

    manifest.py   the frozen test set (stable task ids), read from a generation run
    judge.py      one LLM verdict per success condition → success + partial credit
    score.py      judgements → `evaluation.json`, in the same shape inference runs write
"""
