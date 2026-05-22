"""
Agentic Multi-Agent System (MAS) for the AI Visualization Engine.
Implements a Supervisor-Worker pattern to route tasks and manage tool hallucinations.
"""

import sys
import traceback
from typing import Any

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

# The Supervisor's specialized pseudo-tool used to route tasks to workers.
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
    """
    Fetch available tools from the MCP server and filter them based on the agent's role.

    Args:
        session: An active MCP ClientSession.
        allowed_names: A list of tool names this specific agent is allowed to see.

    Returns:
        A list of dictionaries representing the scoped tools for the Ollama API.
    """
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
    max_iterations: int = 4
) -> str:
    """
    Executes a specialist agent's loop. Handles retries if MCP tool execution fails.
    """
    allowed_tools = AGENT_TOOLS.get(agent_role, [])
    tools = await _get_mcp_tools(session, allowed_names=allowed_tools)

    system_prompt = WORKER_PROMPTS.get(agent_role, "You are a helpful assistant.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task_instruction}"}
    ]

    global_logs.append(("info", f"Supervisor delegated task to '{agent_role}' agent."))

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
            global_logs.append(("error", error_msg))
            return error_msg

        # If the worker didn't call any tools, its task is complete.
        if not response["message"].get("tool_calls"):
            worker_summary = response["message"].get("content", f"{agent_role} agent completed task silently.")
            global_logs.append(("info", f"Worker '{agent_role}' finished task successfully."))
            return worker_summary

        tool_calls = response["message"]["tool_calls"]
        mcp_tool_results: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]
            tool_args["data_file_path"] = data_file_path  # Inject file path

            try:
                result = await session.call_tool(tool_name, arguments=tool_args)
                
                tool_output_raw = ""
                if result.content and isinstance(result.content[0], types.TextContent):
                    tool_output_raw = result.content[0].text

                # Handle Execution Errors (LLM Retry Prompting)
                if "Error:" in tool_output_raw:
                    global_logs.append(("warning", f"Worker '{agent_role}' tool '{tool_name}' failed: {tool_output_raw}. Retrying..."))
                    mcp_tool_results.append({
                        "role": "tool",
                        "content": f"Execution Error: {tool_output_raw}\nPlease correct your code/parameters and try again.",
                    })
                
                # Handle Data Summaries & Stats
                elif tool_name in ["get_column_summary", "run_correlation", "run_group_comparison", "run_linear_regression"]:
                    mcp_tool_results.append({
                        "role": "tool",
                        "content": tool_output_raw,
                    })
                
                # Handle Plotting Tools
                else:
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
                        global_plots.append({
                            "path": tool_output_raw.strip(),
                            "code": "# Source code unavailable.",
                            "name": tool_name
                        })
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": f"Successfully generated plot at {tool_output_raw.strip()}",
                        })

            except Exception as e:
                error_msg = f"Tool '{tool_name}' crashed: {str(e)}"
                global_logs.append(("error", error_msg))
                mcp_tool_results.append({
                    "role": "tool",
                    "content": error_msg,
                })

        # Append tool execution results back to worker's context
        messages.extend(mcp_tool_results)

    # If the loop maxes out
    return f"Worker '{agent_role}' reached maximum iterations and terminated. Last known state appended."


async def run_analysis(
    messages: list[dict[str, Any]],
    data_file_path: str,
    model_name: str,
    mcp_server_script: str,
    max_iterations: int = 7
) -> VizAnalysisResult:
    """
    Main entrypoint: Runs the Supervisor Agent which delegates to workers.
    """
    server_params = StdioServerParameters(
        command="python3",
        args=[mcp_server_script],
    )

    plot_results: list[PlotArtifact] = []
    logs: list[tuple[str, str]] = []
    
    # Inject Supervisor Prompt into the incoming conversation
    supervisor_messages = [{"role": "system", "content": SUPERVISOR_PROMPT}] + messages

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # The supervisor ONLY has access to the delegate_task pseudo-tool
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
                        logs.append(("error", f"Error communicating with Supervisor: {e}"))
                        break

                    # If supervisor decides it is done (no tool calls), exit loop
                    if not response["message"].get("tool_calls"):
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

                            # Execute the worker agent
                            worker_result = await _run_worker_agent(
                                session=session,
                                agent_role=agent_role,
                                task_instruction=task_instruction,
                                data_file_path=data_file_path,
                                model_name=model_name,
                                global_plots=plot_results,
                                global_logs=logs
                            )

                            supervisor_tool_results.append({
                                "role": "tool",
                                "content": f"Results from {agent_role}:\n{worker_result}",
                            })

                    # Feed worker results back to the supervisor
                    supervisor_messages.extend(supervisor_tool_results)

    except Exception as e:
        logs.append(("error", f"Fatal error in MAS session: {str(e)}\n{traceback.format_exc()}"))

    # Extract the final synthesis summary from the Supervisor
    summary = ""
    if supervisor_messages and supervisor_messages[-1].get("role") == "assistant":
        summary = supervisor_messages[-1].get("content", "")
        
    if not summary:
        summary = "Analysis complete. Please review the generated visualizations below."

    # Enforce type safety
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