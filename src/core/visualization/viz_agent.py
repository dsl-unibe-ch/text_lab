"""
Agentic Multi-Agent System (MAS) for the AI Visualization Engine.
Implements a Supervisor-Worker pattern to route tasks and manage tool hallucinations.
"""

import sys
import traceback
from typing import Any, Callable

import ollama
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from core.visualization.viz_config import (
    AGENT_TOOLS,
    INTERACTIVE_PROMPT,
    STATIC_PROMPT,
    STATS_PROMPT,
    SUPERVISOR_PROMPT,
    PlotArtifact,
    VizAnalysisResult,
)

WORKER_PROMPTS = {
    "interactive": INTERACTIVE_PROMPT,
    "static": STATIC_PROMPT,
    "stats": STATS_PROMPT,
}

DELEGATE_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate_task",
        "description": "Delegate a sub-task to a specialist agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_role": {
                    "type": "string",
                    "enum": ["interactive", "static", "stats"],
                    "description": "The specific agent to assign the task to."
                },
                "task_instruction": {
                    "type": "string",
                    "description": "Clear instructions on what the agent should do."
                }
            },
            "required": ["agent_role", "task_instruction"]
        }
    }
}


async def _get_mcp_tools(session: ClientSession, allowed_names: list[str] | None = None) -> list[dict[str, Any]]:
    tool_list_response = await session.list_tools()
    ollama_tools: list[dict[str, Any]] = []
    
    for tool in tool_list_response.tools:
        if allowed_names is None or tool.name in allowed_names:
            ollama_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
            )
            
    return ollama_tools


async def _run_worker_agent(
    session: ClientSession,
    agent_role: str,
    task_instruction: str,
    data_file_path: str,
    model_name: str,
    global_plots: list[PlotArtifact],
    global_logs: list[tuple[str, str]],
    max_iterations: int = 4,
    log_callback: Callable[[str, str], None] | None = None
) -> str:
    def _log(l_type: str, msg: str):
        global_logs.append((l_type, msg))
        if log_callback:
            log_callback(l_type, msg)

    allowed_tools = AGENT_TOOLS.get(agent_role, [])
    tools = await _get_mcp_tools(session, allowed_names=allowed_tools)

    system_prompt = WORKER_PROMPTS.get(agent_role, "You are a helpful assistant.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task_instruction}"}
    ]

    _log("info", f"Supervisor delegated task to '{agent_role}' agent.")

    for iteration in range(max_iterations):
        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=tools,
            )
            messages.append(response["message"])
        except Exception as e:
            error_msg = f"Worker '{agent_role}' failed to communicate with Ollama: {e}"
            _log("error", error_msg)
            return error_msg

        if not response["message"].get("tool_calls"):
            worker_summary = response["message"].get("content", f"{agent_role} agent completed task silently.")
            _log("info", f"Worker '{agent_role}' finished task successfully.")
            return worker_summary

        tool_calls = response["message"]["tool_calls"]
        mcp_tool_results: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]
            tool_args["data_file_path"] = data_file_path

            try:
                result = await session.call_tool(tool_name, arguments=tool_args)
                
                tool_output_raw = ""
                if result.content and isinstance(result.content[0], types.TextContent):
                    tool_output_raw = result.content[0].text
                
                # Handle Data Summaries & Stats
                if tool_name in ["get_column_summary", "run_correlation", "run_group_comparison", "run_linear_regression"]:
                    if "Error" in tool_output_raw:
                        _log("warning", f"Worker '{agent_role}' stat tool '{tool_name}' failed. Retrying...")
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": f"Execution Error: {tool_output_raw.strip()}\nPlease correct your code/parameters and try again.",
                        })
                    else:
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": tool_output_raw,
                        })
                
                # Handle Plotting Tools
                else:
                    # FIX: Plotting tools MUST return path|||code. If they don't, force a retry.
                    if "|||" in tool_output_raw:
                        path_part, code_part = tool_output_raw.split("|||", 1)
                        global_plots.append({
                            "path": path_part.strip(),
                            "code": code_part.strip(),
                            "name": tool_name
                        })
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": f"Successfully generated plot at {path_part.strip()}",
                        })
                    else:
                        _log("warning", f"Worker '{agent_role}' plot tool '{tool_name}' failed: {tool_output_raw.strip()}. Retrying...")
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": f"Execution Error: {tool_output_raw.strip()}\nPlease correct your code or parameters and try again.",
                        })

            except Exception as e:
                error_msg = f"Tool '{tool_name}' crashed: {str(e)}"
                _log("error", error_msg)
                mcp_tool_results.append({
                    "role": "tool",
                    "content": error_msg,
                })

        messages.extend(mcp_tool_results)

    _log("error", f"Worker '{agent_role}' reached maximum iterations without resolving issues.")
    return f"Worker '{agent_role}' reached maximum iterations and terminated. Last known state appended."


async def run_analysis(
    messages: list[dict[str, Any]],
    data_file_path: str,
    model_name: str,
    mcp_server_script: str,
    max_iterations: int = 7,
    log_callback: Callable[[str, str], None] | None = None
) -> VizAnalysisResult:
    def _log(l_type: str, msg: str):
        logs.append((l_type, msg))
        if log_callback:
            log_callback(l_type, msg)

    server_params = StdioServerParameters(
        command="python3",
        args=[mcp_server_script],
    )

    plot_results: list[PlotArtifact] = []
    logs: list[tuple[str, str]] = []
    
    supervisor_messages = [{"role": "system", "content": SUPERVISOR_PROMPT}] + messages

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                supervisor_tools = [DELEGATE_TASK_TOOL]

                for iteration in range(max_iterations):
                    try:
                        response = ollama.chat(
                            model=model_name,
                            messages=supervisor_messages,
                            tools=supervisor_tools,
                        )
                        supervisor_messages.append(response["message"])
                    except Exception as e:
                        _log("error", f"Error communicating with Supervisor: {e}")
                        break

                    if not response["message"].get("tool_calls"):
                        _log("info", "Supervisor completed task delegation and synthesized final summary.")
                        break

                    tool_calls = response["message"]["tool_calls"]
                    supervisor_tool_results: list[dict[str, Any]] = []

                    for tool_call in tool_calls:
                        if tool_call["function"]["name"] == "delegate_task":
                            agent_role = tool_call["function"]["arguments"].get("agent_role")
                            task_instruction = tool_call["function"]["arguments"].get("task_instruction")

                            if agent_role not in WORKER_PROMPTS:
                                supervisor_tool_results.append({
                                    "role": "tool",
                                    "content": f"Error: Agent role '{agent_role}' does not exist.",
                                })
                                continue

                            worker_result = await _run_worker_agent(
                                session=session,
                                agent_role=agent_role,
                                task_instruction=task_instruction,
                                data_file_path=data_file_path,
                                model_name=model_name,
                                global_plots=plot_results,
                                global_logs=logs,
                                log_callback=log_callback
                            )

                            supervisor_tool_results.append({
                                "role": "tool",
                                "content": f"Results from {agent_role}:\n{worker_result}",
                            })

                    supervisor_messages.extend(supervisor_tool_results)

    except Exception as e:
        _log("error", f"Fatal error in MAS session: {str(e)}\n{traceback.format_exc()}")

    summary = ""
    if supervisor_messages and supervisor_messages[-1].get("role") == "assistant":
        summary = supervisor_messages[-1].get("content", "")
        
    if not summary:
        summary = "Analysis complete. Please review the generated visualizations below."

    final_logs: list[tuple[Any, str]] = []
    for log_type, msg in logs:
        if log_type in ("info", "warning", "error"):
            final_logs.append((log_type, msg))
        else:
            final_logs.append(("info", msg))

    return {
        "summary": summary,
        "plots": plot_results,
        "logs": final_logs
    }