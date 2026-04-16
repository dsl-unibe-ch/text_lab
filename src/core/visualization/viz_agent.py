"""
Agentic loop for the AI Visualization Engine.
Handles communication between the Ollama LLM and the MCP server.
"""

import sys
import traceback
from typing import Any

import ollama
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from core.visualization.viz_config import PlotArtifact, VizAnalysisResult


async def _get_mcp_tools(session: ClientSession) -> list[dict[str, Any]]:
    """
    Fetch available tools from the MCP server and format them for Ollama.

    Args:
        session: An active MCP ClientSession.

    Returns:
        A list of dictionaries representing the tools in the format expected
        by the Ollama API.
    """
    tool_list_response = await session.list_tools()
    ollama_tools: list[dict[str, Any]] = []
    
    for tool in tool_list_response.tools:
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


async def run_analysis(
    messages: list[dict[str, Any]],
    data_file_path: str,
    model_name: str,
    mcp_server_script: str,
    max_iterations: int = 5
) -> VizAnalysisResult:
    """
    Run the LLM analysis in an agentic loop, allowing the AI to call tools,
    evaluate data, and generate plots autonomously.

    Args:
        messages: The conversation history, including system and user prompts.
        data_file_path: The absolute path to the data file to analyze.
        model_name: The Ollama model to use for generation.
        mcp_server_script: The path to the MCP server Python script.
        max_iterations: The maximum number of tool-calling iterations allowed
            before forcing a final summary (prevents infinite loops).

    Returns:
        A VizAnalysisResult dictionary containing the markdown summary,
        a list of generated PlotArtifacts, and execution logs.
    """
    server_params = StdioServerParameters(
        command="python3",
        args=[mcp_server_script],
    )

    plot_results: list[PlotArtifact] = []
    logs: list[tuple[str, str]] = []

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await _get_mcp_tools(session)

                for iteration in range(max_iterations):
                    # 1. Ask Ollama for the next action
                    try:
                        response = ollama.chat(
                            model=model_name,
                            messages=messages,
                            tools=tools,
                        )
                        messages.append(response["message"])
                    except Exception as e:
                        logs.append(("error", f"Error communicating with Ollama: {e}"))
                        break

                    # 2. Check if the AI decided to stop calling tools
                    if not response["message"].get("tool_calls"):
                        break

                    # 3. Execute the requested tools via MCP
                    tool_calls = response["message"]["tool_calls"]
                    mcp_tool_results: list[dict[str, Any]] = []

                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        tool_args = tool_call["function"]["arguments"]
                        
                        # Inject the data path automatically so the LLM does not have to guess it
                        tool_args["data_file_path"] = data_file_path

                        try:
                            result = await session.call_tool(tool_name, arguments=tool_args)

                            tool_output_raw = ""
                            if result.content and isinstance(result.content[0], types.TextContent):
                                tool_output_raw = result.content[0].text

                            # Handle Tool Errors gracefully
                            if "Error:" in tool_output_raw:
                                logs.append(("warning", f"Tool '{tool_name}' failed: {tool_output_raw}"))
                                mcp_tool_results.append({
                                    "role": "tool",
                                    "content": f"Tool Error: {tool_output_raw}",
                                })
                            
                            # Handle Data Summary Tools (Return text context to the LLM)
                            elif tool_name == "get_column_summary":
                                mcp_tool_results.append({
                                    "role": "tool",
                                    "content": tool_output_raw,
                                })
                                
                            # Handle Plotting Tools (Parse the path and code snippet)
                            else:
                                if "|||" in tool_output_raw:
                                    path_part, code_part = tool_output_raw.split("|||", 1)
                                    plot_results.append({
                                        "path": path_part.strip(),
                                        "code": code_part.strip(),
                                        "name": tool_name
                                    })
                                    mcp_tool_results.append({
                                        "role": "tool",
                                        "content": f"Successfully generated plot at {path_part.strip()}",
                                    })
                                else:
                                    plot_results.append({
                                        "path": tool_output_raw.strip(),
                                        "code": "# Source code unavailable for this plot.",
                                        "name": tool_name
                                    })
                                    mcp_tool_results.append({
                                        "role": "tool",
                                        "content": f"Successfully generated plot at {tool_output_raw}",
                                    })

                        except Exception as e:
                            error_msg = f"Failed to execute tool '{tool_name}': {str(e)}"
                            logs.append(("error", error_msg))
                            mcp_tool_results.append({
                                "role": "tool",
                                "content": error_msg,
                            })

                    # 4. Append tool results to history to inform the AI's next decision
                    messages.extend(mcp_tool_results)

    except Exception as e:
        logs.append(("error", f"Fatal error in MCP session: {str(e)}\n{traceback.format_exc()}"))

    # 5. Extract the final conversational summary
    summary = ""
    if messages and messages[-1].get("role") == "assistant":
        summary = messages[-1].get("content", "")
        
    if not summary:
        summary = "Analysis complete. Please review the generated visualizations below."

    # Enforce type safety before returning
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