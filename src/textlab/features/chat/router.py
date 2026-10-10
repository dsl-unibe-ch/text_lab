"""Deciding whether a chat message needs the data-analysis agents.

With a table attached, the model is asked (as a tool call) whether the
user wants charts or statistics; if so, the Visualization feature's agents
answer instead of the chat model.
"""

from __future__ import annotations

import json

from textlab.common.ollama import chat_no_think

# The router makes a yes/no decision, which should be predictable: the same
# question must not route to analysis one time and to plain chat the next.
# Temperature does not affect speed and does not make Ollama reload the model.
ROUTER_TEMPERATURE = 0.1

ROUTER_SYSTEM_PROMPT = (
    "You are a routing supervisor for a chat assistant. The user is chatting "
    "and has "
    "uploaded a tabular dataset (a schema is provided below). Decide whether "
    "answering "
    "the user's latest message requires generating plots/charts or running "
    "statistical "
    "analysis (correlations, t-tests, ANOVA, regression) on that dataset.\n\n"
    "- If the request requires visualisations or statistics on the dataset, "
    "call the "
    "`analyze_data` tool with a clear, self-contained instruction derived "
    "from the user's "
    "request and conversation.\n"
    "- If the request is general conversation, a factual question, or can be "
    "answered from "
    "the dataset text already in context WITHOUT producing plots or "
    "statistical tests, do "
    "NOT call any tool and simply reply normally.\n\n"
    "Only call `analyze_data` when visual or statistical output is genuinely "
    "needed."
)

ANALYZE_DATA_TOOL = {
    "type": "function",
    "function": {
        "name": "analyze_data",
        "description": (
            "Run data visualisation and/or statistical analysis on the user's "
            "uploaded "
            "dataset. Use this only when the user's request requires plots, "
            "charts, or "
            "statistical tests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": (
                        "A clear, self-contained analysis instruction derived "
                        "from the "
                        "user's request (e.g. 'Plot an interactive scatter of "
                        "age vs income "
                        "and run a correlation between them')."
                    ),
                }
            },
            "required": ["instruction"],
        },
    },
}


def decide_tool_use(
    model_name: str,
    user_text: str,
    schema_text: str,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[bool, str]:
    """Ask the model whether a message needs the data-analysis tools.

    The model acts as a router: it decides whether answering the user's
    latest message requires plots or statistics on the uploaded dataset.

    Args:
        model_name: The Ollama model to use for routing. Must support tool
            calling.
        user_text: The user's latest message.
        schema_text: A compact schema/summary of the uploaded dataset.
        chat_history: Prior conversation turns (excluding the current user
            turn).

    Returns:
        tuple[bool, str]: (use_tools, instruction). When use_tools is False the
        instruction is an empty string. On any error, falls back to (False,
        "").
    """
    history = chat_history or []
    router_messages = (
        [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        + history
        + [
            {
                "role": "user",
                "content": (
                    f"Dataset schema:\n{schema_text}\n\n"
                    f"User message: {user_text}"
                ),
            }
        ]
    )

    try:
        response = chat_no_think(
            model=model_name,
            messages=router_messages,
            tools=[ANALYZE_DATA_TOOL],
            options={"temperature": ROUTER_TEMPERATURE},
        )
    except Exception:
        # Router failed (e.g. model lacks tool support) — fall back to plain
        # chat.
        return False, ""

    message = (
        response["message"] if isinstance(response, dict) else response.message
    )
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not tool_calls:
        return False, ""

    for tool_call in tool_calls:
        fn = (
            tool_call["function"]
            if isinstance(tool_call, dict)
            else tool_call.function
        )
        name = fn["name"] if isinstance(fn, dict) else fn.name
        if name != "analyze_data":
            continue
        args = fn["arguments"] if isinstance(fn, dict) else fn.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        instruction = (args or {}).get("instruction", "").strip()
        if not instruction:
            instruction = user_text.strip()
        return True, instruction

    return False, ""
