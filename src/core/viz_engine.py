import os
import pandas as pd
from io import BytesIO
import ollama
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

# --- System & Default Prompts ---
SYSTEM_PROMPT = """
You are an expert data analyst. Your task is to generate visualisations based on a user's request and the first 5 rows of their dataset.

1.  **Standard Tools:** You have access to a suite of specific plotting tools:
    * `plot_histogram`, `plot_countplot`, `plot_scatterplot`
    * `plot_boxplot`, `plot_violinplot`, `plot_lineplot`
    * `plot_correlation_heatmap`, `plot_pairplot`
    Use these for standard requests (e.g., "Show me the distribution of Age").

2.  **Custom Code Tool (`generate_custom_plot`):** * Use this tool for complex requests.
    * **CRITICAL DATA TYPE RULE:** The data is loaded from CSV/Excel. Columns that look like dates or numbers might be loaded as Strings (Objects).
    * **YOU MUST CONVERT DATA TYPES EXPLICITLY.** * If you need to plot a date, running `df['date'] = pd.to_datetime(df['date'], errors='coerce')` is MANDATORY before using `.dt` accessors.
      * If you need to plot a number, run `pd.to_numeric(..., errors='coerce')` first.
    * Do NOT rely on pandas auto-detection.
    * The data is already loaded into `df`. Do NOT write code to load the file.

3.  The user will provide a prompt and the `head()` of their data.

4.  **CRITICAL:** All tools require a `data_file_path` argument. You DO NOT need to provide this. It will be injected for you.

5.  Call multiple tools if it makes sense.

6.  After you call the tools, you will receive their output. Provide a final, single response summarizing what you did in Markdown.
"""

DEFAULT_PROMPT = "Please perform a basic exploratory data analysis. Generate a few useful plots to understand the data's distribution and relationships."

# --- Data Helpers ---

def parse_dataframe(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Parses raw bytes into a Pandas DataFrame based on file extension."""
    try:
        if file_name.endswith(".csv"):
            return pd.read_csv(BytesIO(file_bytes), encoding="latin1")
        elif file_name.endswith(".tsv"):
            return pd.read_csv(BytesIO(file_bytes), sep="\t", encoding="latin1")
        elif file_name.endswith((".xls", ".xlsx")):
            return pd.read_excel(BytesIO(file_bytes))
        elif file_name.endswith(".json"):
            return pd.read_json(BytesIO(file_bytes))
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
    except Exception as e:
        raise RuntimeError(f"Error reading file: {e}")

def save_data_file(file_bytes: bytes, file_name: str, run_dir: str) -> str:
    """Saves the uploaded file to the temporary workspace."""
    file_extension = os.path.splitext(file_name)[1]
    data_file_path = os.path.join(run_dir, f"uploaded_data{file_extension}")
    with open(data_file_path, "wb") as f:
        f.write(file_bytes)
    return data_file_path

# --- MCP Client Logic ---

async def get_mcp_tools(session: ClientSession) -> list:
    tool_list_response = await session.list_tools()
    ollama_tools = []
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

async def run_analysis(messages, data_file_path, model_name, mcp_server_script):
    """
    Runs the LLM analysis and triggers MCP tools.
    Returns: (summary_markdown, list_of_plots, list_of_logs)
    """
    server_params = StdioServerParameters(
        command="python3",
        args=[mcp_server_script],
    )

    plot_results = []
    summary = ""
    logs = [] # Format: [("type", "message")] to pass back to UI safely

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await get_mcp_tools(session)

            # 1. First call to Ollama
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=messages,
                    tools=tools,
                )
                messages.append(response["message"])
            except Exception as e:
                logs.append(("error", f"Error calling Ollama: {e}"))
                return "Failed to get analysis from LLM.", [], logs

            # 2. Check for and execute tool calls
            if response["message"].get("tool_calls"):
                tool_calls = response["message"]["tool_calls"]
                mcp_tool_results = []

                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]
                    tool_args["data_file_path"] = data_file_path

                    try:
                        # 3. Execute tool
                        result = await session.call_tool(tool_name, arguments=tool_args)

                        tool_output_raw = ""
                        if result.content and isinstance(result.content[0], types.TextContent):
                            tool_output_raw = result.content[0].text

                        if "Error:" in tool_output_raw:
                            logs.append(("warning", f"Tool '{tool_name}' failed: {tool_output_raw}"))
                            mcp_tool_results.append({
                                "role": "tool",
                                "content": f"Tool Error: {tool_output_raw}",
                            })
                        else:
                            # Parse result delimiter (Path ||| Code)
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
                                    "code": "# Code transparency not available for this plot.",
                                    "name": tool_name
                                })
                                mcp_tool_results.append({
                                    "role": "tool",
                                    "content": f"Successfully generated plot at {tool_output_raw}",
                                })

                    except Exception as e:
                        logs.append(("error", f"Failed to execute tool '{tool_name}': {e}"))
                        mcp_tool_results.append({
                            "role": "tool",
                            "content": f"Failed to execute tool: {str(e)}",
                        })

                # 4. Send results back to Ollama
                messages.extend(mcp_tool_results)
                try:
                    final_response = ollama.chat(model=model_name, messages=messages, tools=tools)
                    summary = final_response["message"]["content"]
                except Exception as e:
                    logs.append(("error", f"Error getting final summary: {e}"))
                    summary = "Failed to generate summary, but plots were created."

            else:
                summary = response["message"]["content"]

    return summary, plot_results, logs