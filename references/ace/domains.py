"""The benchmark-bound clauses of ACE's reflector and curator prompts.

ACE's prompts are written for AppWorld: "You are an expert AppWorld coding agent", the
generator's output is called "generated code", the schema note names
`apis.blah.show_contents()`, and both prompts carry two worked examples about Venmo
roommates and Spotify pagination. Those clauses — and only those — are re-written per
benchmark here, the same way `references/reasoningbank/domains.py` and
`references/expel/extraction.py` re-write the clauses their own papers re-write per
domain. Every other word of every prompt is ACE's.

The `appworld` entry is ACE's own text, verbatim.
"""

from __future__ import annotations

_DOMAINS: dict[str, dict[str, str]] = {
    "appworld": {
        "expert": "AppWorld coding agent",
        "artifact": "generated code",
        "attempt_label": "Current Generated Attempt (latest attempt, with reasoning and planning)",
        "schema_note": (
            "Explicitly curate the output format/schema of APIs used when unclear or "
            "mismatched with expectations (e.g., `apis.blah.show_contents()` returns a list "
            "of content_ids (strings), not content objects)"
        ),
        "reflector_examples": """**Example 1:**
Generated Code: [Code that tries to identify roommates by parsing Venmo transaction descriptions using keywords like "rent", "utilities"]

Response:
{
  "reasoning": "From the generated trajectory, the agent attempted to infer who the 'roommates' are by scanning Venmo transaction descriptions for keywords (e.g., 'rent'). This approach is fragile because descriptions are user-written, non-standard, and can omit or mislead relationship semantics. The Phone app is the authoritative source for contact relationships, and should be used to precisely identify 'roommates' before filtering transactions.",
  "error_identification": "Identity resolution relied on heuristic keyword parsing in transaction descriptions instead of querying the authoritative Phone contacts.",
  "root_cause_analysis": "The agent implicitly assumed that transactional free-form text is a reliable proxy for relationships, ignoring that contact relationships are normalized elsewhere (Phone app).",
  "correct_approach": "Authenticate with the Phone app, call apis.phone.search_contacts() to retrieve contacts with relationship 'roommate', then use their precise identifiers (emails/phone numbers) to filter Venmo transactions.",
  "key_insight": "Resolve identities from the correct source app (Phone contacts) rather than inferring from indirect, free-form text."
}

**Example 2:**
Generated Code: [Code that uses for i in range(10) to paginate through playlists, then calls complete_task() with the count]

Response:
{
  "reasoning": "Analyzing the generated code, I notice the agent used 'for i in range(10)' for pagination when calling the playlist API. This is problematic because the agent has no way of knowing whether there are exactly 10 pages or fewer. Using a fixed range means the agent could miss data on pages beyond the 10th page. The proper approach for pagination is to continue until the API returns empty results.",
  "error_identification": "The pagination logic used an arbitrary fixed limit (range(10)) instead of continuing until all available data is collected.",
  "root_cause_analysis": "The agent prioritized avoiding infinite loops over ensuring complete data collection, choosing a 'safe' fixed upper bound without considering that this could lead to incomplete results when the actual data exceeds the arbitrary limit.",
  "correct_approach": "Use while True loop with proper break condition: continue calling the API with incrementing page_index until the API returns empty results, null, or an empty list, then break the loop.",
  "key_insight": "For pagination, always use while True loop with proper termination conditions instead of fixed range iterations to ensure complete data collection across all available pages."
}""",
        "curator_examples": """**Example 1:**
Task Context: "Find money sent to roommates since Jan 1 this year"
Current Playbook: [Basic API usage guidelines]
Generated Attempt: [Code that failed because it used transaction descriptions to identify roommates instead of Phone contacts]
Reflections: "The agent failed because it tried to identify roommates by parsing Venmo transaction descriptions instead of using the Phone app's contact relationships. This led to incorrect identification and wrong results."

Response:
{
  "reasoning": "The reflection shows a critical error where the agent used unreliable heuristics (transaction descriptions) instead of the authoritative source (Phone app contacts) to identify relationships. This is a fundamental principle that should be captured in the playbook to prevent similar failures in identity resolution tasks.",
  "operations": [
    {
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "Always resolve identities from the correct source app\\n- When you need to identify relationships (roommates, contacts, etc.), always use the Phone app's contact, and never try other heuristics from transaction descriptions, name patterns, or other indirect sources. These heuristics are unreliable and will cause incorrect results."
    }
  ]
}

**Example 2:**
Task Context: "Count all playlists in Spotify"
Current Playbook: [Basic authentication and API calling guidelines]
Generated Attempt: [Code that used for i in range(10) loop and missed playlists on later pages]
Reflections: "The agent used a fixed range loop for pagination instead of properly iterating through all pages until no more results are returned. This caused incomplete data collection."

Response:
{
  "reasoning": "The reflection identifies a pagination handling error where the agent used an arbitrary fixed range instead of proper pagination logic. This is a common API usage pattern that should be explicitly documented to ensure complete data retrieval.",
  "operations": [
    {
      "type": "ADD",
      "section": "apis_to_use_for_specific_information",
      "content": "About pagination: many APIs return items in \\"pages\\". Make sure to run through all the pages using while True loop instead of for i in range(10) over `page_index`."
    }
  ]
}""",
    },
    "tau2": {
        "expert": "customer-service agent and educator",
        "artifact": "conversation and tool calls",
        "attempt_label": "Current Attempt (the latest conversation, with the agent's reasoning and tool calls)",
        "schema_note": (
            "Explicitly curate what a tool actually returns, and what the domain policy "
            "actually requires, when either is unclear or mismatched with expectations "
            "(e.g. a lookup tool matches exact field values only, so a partial or "
            "plausible-looking argument returns `User not found` rather than a near match)"
        ),
        "reflector_examples": """**Example 1:**
Attempt: [The agent called the order-modification tool using the item name the user typed, and the tool returned an error]

Response:
{
  "reasoning": "The agent passed the user's own wording ('the blue one') where the tool expects an item id. The tools in this domain match exact stored values, so a paraphrase never partially matches — it returns an error that carries no information about the right value. The agent should first list the order's items, read the id off the record, and pass that.",
  "error_identification": "A tool argument was filled from the user's phrasing instead of from a value read out of the domain's own records.",
  "root_cause_analysis": "The agent treated the conversation as the source of truth for identifiers, when the records are.",
  "correct_approach": "Look the order up, read the exact item id from the returned record, then call the modification tool with that id.",
  "key_insight": "Fill tool arguments from values the environment returned, never from the user's wording."
}

**Example 2:**
Attempt: [The agent cancelled an order immediately after the user asked, without the confirmation the policy requires]

Response:
{
  "reasoning": "The domain policy requires an explicit confirmation before an irreversible action. The agent carried out the cancellation on the first request. Even though the user did want it, skipping the required confirmation step is itself the failure, because the policy — not the user's intent — governs what the agent may do.",
  "error_identification": "An irreversible action was taken without the confirmation the policy requires first.",
  "root_cause_analysis": "The agent optimised for satisfying the request in as few turns as possible and treated the policy's confirmation step as a formality.",
  "correct_approach": "State what will happen, ask for explicit confirmation, wait for the user's reply, and only then call the tool.",
  "key_insight": "Before any irreversible action, complete every step the policy requires — confirmation included — even when the user's intent is already clear."
}""",
        "curator_examples": """**Example 1:**
Task Context: "Exchange the blue shirt in my last order for the red one"
Current Playbook: [Basic tool usage guidelines]
Generated Attempt: [The agent passed the user's wording as an item id and the tool errored]
Reflections: "The agent filled a tool argument from the user's phrasing instead of from a value read out of the order record."

Response:
{
  "reasoning": "The reflection shows the agent treating the conversation as the source of truth for identifiers. The tools match exact stored values, so this fails with no partial match to learn from. That is a general rule worth stating once in the playbook.",
  "operations": [
    {
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "Fill every tool argument from a value the environment returned, never from the user's wording. Look the record up first and read the exact identifier off it — these tools match exact field values, so a paraphrase returns a plain not-found rather than a near match."
    }
  ]
}

**Example 2:**
Task Context: "Cancel order #W123"
Current Playbook: [Basic tool usage guidelines]
Generated Attempt: [The agent cancelled on the first request, with no confirmation step]
Reflections: "The policy requires an explicit confirmation before an irreversible action; the agent skipped it."

Response:
{
  "reasoning": "The failure is procedural rather than factual: the action was right, the required preceding step was missing. A checklist item is the right shape for it.",
  "operations": [
    {
      "type": "ADD",
      "section": "verification_checklist",
      "content": "Before any irreversible action (cancel, exchange, refund, address change), state what will happen and get an explicit confirmation from the user, then act. The policy requires this even when the user's intent is already clear."
    }
  ]
}""",
    },
    "automationbench": {
        "expert": "business-operations agent and educator",
        "artifact": "tool calls and their results",
        "attempt_label": "Current Attempt (the latest trajectory, with the agent's reasoning and tool calls)",
        "schema_note": (
            "Explicitly curate the request and response shape of the endpoints used when "
            "unclear or mismatched with expectations (e.g. a list endpoint returns one page "
            "and a cursor rather than the whole collection, so a single call under-reports)"
        ),
        "reflector_examples": """**Example 1:**
Attempt: [The agent answered "which vendors are overdue" from the first page of the invoices endpoint]

Response:
{
  "reasoning": "The agent called the invoice listing endpoint once and treated the result as the whole collection. These endpoints are paginated, so the answer covered only the first page and silently omitted the rest. The request named a whole set, so every page has to be read before the set can be filtered.",
  "error_identification": "A question about a whole set was answered from a single unpaginated read.",
  "root_cause_analysis": "The agent assumed a list endpoint returns everything, because the first response looked complete on its own.",
  "correct_approach": "Follow the cursor (or increment the page parameter) until the endpoint returns no further items, collect the union, and only then filter and report.",
  "key_insight": "A request about a whole set requires reading every page of every list endpoint it depends on before filtering."
}

**Example 2:**
Attempt: [The agent posted the update and finished without re-reading the record]

Response:
{
  "reasoning": "The write returned without an error and the agent treated that as proof the effect landed. A 2xx on these endpoints only says the request was accepted in the shape sent — it does not confirm the field holds the intended value, and this task is graded on the final state of the world.",
  "error_identification": "A write was assumed to have taken effect because the call did not error.",
  "root_cause_analysis": "The agent conflated a successful call with a successful outcome.",
  "correct_approach": "After a write, read the record back and check the field actually carries the intended value before finishing.",
  "key_insight": "A non-erroring write is not evidence the state changed — read it back and confirm before completing."
}""",
        "curator_examples": """**Example 1:**
Task Context: "Tell me which vendors are overdue this month"
Current Playbook: [Basic endpoint usage guidelines]
Generated Attempt: [The agent answered from the first page of the invoices endpoint]
Reflections: "A question about a whole set was answered from a single unpaginated read."

Response:
{
  "reasoning": "This is the standard pagination failure and it is invisible in the output — the answer looks complete. Worth stating once, as a hard rule rather than a checklist item, because it changes how every list read is written.",
  "operations": [
    {
      "type": "ADD",
      "section": "strategies_and_hard_rules",
      "content": "When a request concerns a whole set (all vendors, every certificate, each overdue invoice), read EVERY page of every list endpoint it depends on before filtering. These endpoints return one page plus a cursor; a single call silently under-reports and the answer still looks complete."
    }
  ]
}

**Example 2:**
Task Context: "Update the renewal date on the Acme contract row"
Current Playbook: [Basic endpoint usage guidelines]
Generated Attempt: [The agent posted the update and finished without re-reading]
Reflections: "A write was assumed to have taken effect because the call did not error."

Response:
{
  "reasoning": "Grading is a final-state assertion sweep, so an accepted request that did not land scores as a miss. A verification step is the right shape.",
  "operations": [
    {
      "type": "ADD",
      "section": "verification_checklist",
      "content": "After every write, read the record back and confirm the field carries the intended value. A non-erroring call only means the request was accepted in the shape sent, not that the state changed."
    }
  ]
}""",
    },
}

CLAUSE_KEYS = (
    "expert",
    "artifact",
    "attempt_label",
    "schema_note",
    "reflector_examples",
    "curator_examples",
)


def domain(benchmark: str) -> dict[str, str]:
    """The prompt clauses for a benchmark."""
    try:
        return _DOMAINS[benchmark]
    except KeyError:
        raise ValueError(
            f"ACE has no prompt clauses for benchmark {benchmark!r}; known: "
            f"{sorted(_DOMAINS)}. Add one (four phrases and two worked examples) in "
            f"references/ace/domains.py."
        ) from None
