"""The domain-bound clauses of ReasoningBank's prompts, one entry per benchmark.

The paper's prompts (Fig. 9, 10, 11) are written for web navigation: "You are an expert in
web navigation", "Do not mention specific websites", and a judge that lists the three kinds
of task a *web* agent gets. Those clauses — and only those — are re-written per benchmark
here, the same way the ExpeL implementation next door re-writes the clauses ExpeL itself
re-writes for each of its own domains. Every other word of every prompt is the paper's.

`specifics` deserves a note. The paper's "Do not mention specific websites, queries, or
string contents" tells the extractor not to tie a memory item to the instance it came from
(that WebArena shop, that search string). The analogue here is the task's own data — this
customer, this order id — NOT the app or tool names, which are the environment's stable
mechanics and are exactly what a transferable item needs to name.
"""

from __future__ import annotations

_DOMAINS = {
    "tau2": {
        # Fig. 9 / Fig. 11: "You are an expert in {expert}."
        "expert": "customer-service agents that resolve a user's request through tool calls",
        "specifics": "specific customers, order identifiers, or string contents",
        "generalize_beyond": "specific conversations or exact wording",
        # Fig. 10: "…evaluating the performance of {agent_description}."
        "agent_description": (
            "a customer-service agent. The agent is designed to help a human user resolve a "
            "request by talking to them and calling the domain's tools, under the domain's "
            "policy"
        ),
        "task_types": """1. Information seeking: The user wants to obtain certain information the domain holds, such as the status of an order, the options of a product, or what a policy allows. The agent's response must contain the information the user wants, or explicitly state that the information is not available. Otherwise, e.g. the agent encounters an exception and responds with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions: an answer the agent never actually looked up with a tool is likely wrong.

2. Content modification: The user wants the domain's records changed, such as cancelling an order, exchanging an item, or updating an address. Carefully examine the agent's tool calls and their results to determine whether the change was really applied, and with the values the user asked for. No need to consider the agent's response.

3. Policy-bound handling: The domain's policy forbids what the user asks, or requires a confirmation or a piece of information first. The agent should follow the policy — refuse, ask, or transfer — rather than act. Carrying out a forbidden action is a failure, and so is refusing something the policy permits.""",
    },
    "appworld": {
        "expert": "software agents that complete a user's task by calling app APIs from Python code",
        "specifics": "specific accounts, task inputs, or string contents",
        "generalize_beyond": "specific tasks or exact wording",
        "agent_description": (
            "an app-automation agent. The agent is designed to complete a task for a human "
            "supervisor by writing Python code against the APIs of the supervisor's apps"
        ),
        "task_types": """1. Information seeking: The user wants to obtain certain information from their apps, such as a count, a date, or the contents of a record. The agent must return that information through the task-completion call, or explicitly state that it is not available. Otherwise, e.g. the agent hits an exception and stops with the error content, the task is considered a failure. Besides, be careful about the sufficiency of the agent's actions. For example, when asked for the top items by some quantity, the agent should retrieve ALL the candidates, including every page of a paginated API, and then order them. If the pagination or the ordering is missing, the task is likely to fail.

2. Content modification: The user wants the state of an app changed, such as sending a message, adding an item, or updating a setting. Carefully examine the agent's API calls and their results to determine whether the change was really applied, and with the values the user asked for. No need to consider the agent's final answer.

3. Multi-app tasks: The task spans several apps, or asks for one action per item of a list. Carefully check that EVERY part was carried out, not just the first one — a partially completed task is a failure.""",
    },
    "automationbench": {
        "expert": (
            "business-operations agents that carry out a workplace request by calling the "
            "REST APIs of the company's SaaS applications"
        ),
        "specifics": "specific records, recipients, identifiers, or string contents",
        "generalize_beyond": "specific requests or exact wording",
        "agent_description": (
            "a business-operations agent. The agent is designed to carry out a request that "
            "arrives as a workplace message — an email, a chat message, a ticket — by reading "
            "the company's records through app APIs and making the writes the request asks for"
        ),
        "task_types": """1. Information seeking: The request asks for something the company's records hold, such as which vendor is overdue, how many certificates expire this month, or what a policy worksheet allows. The agent must report that information, or explicitly state that it is not available. Be careful about the sufficiency of the agent's actions: an answer it never actually looked up is likely wrong, and so is one drawn from a single record when the request names a whole set. Reading only the first page of a paginated list is the common way this fails.

2. Content modification: The request asks for the state of an app to be changed, such as sending a message, creating a task or event, updating a row, or applying a label. Examine the agent's API calls AND their results to determine whether the change was really applied, with the values the request asked for. A call that returned without an error does not prove the effect landed — look for a read that confirms it. No need to consider the agent's closing message.

3. Selection under stated rules: The request names filters, exclusions, holds, date windows or preferences that decide which records qualify, often in a worksheet or policy document the agent must read first. Check that the agent read every source that governs the choice and applied each rule as stated — acting on all records when the request named a subset, or applying a requirement to records that never declared it, are both failures. Where several sources disagree, the most recent notice from the authorized sender governs.""",
    },
}


def domain(benchmark: str) -> dict[str, str]:
    """The prompt clauses for a benchmark."""
    try:
        return _DOMAINS[benchmark]
    except KeyError:
        raise ValueError(
            f"ReasoningBank has no prompt clauses for benchmark {benchmark!r}; known: "
            f"{sorted(_DOMAINS)}. Add one (four phrases and a task-type list) in "
            f"references/reasoningbank/domains.py."
        ) from None
