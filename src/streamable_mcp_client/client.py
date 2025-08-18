import asyncio
import json
import logging
import re
import shutil
from typing import AsyncGenerator

from fastmcp import Client
from mcp.types import TextContent, Tool
import datetime

from openai import AsyncOpenAI

class LLMClient:
    @staticmethod
    async def get_stream_response_reasion_and_content(
        messages: list[dict[str, str]], llm_url, api_key
    ) -> AsyncGenerator[str, None]:
        
        client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=llm_url)

        response = await client.chat.completions.create(
            messages=messages, stream=True, model="deepseek-reasoner"
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.reasoning_content:
                yield {"type": "thought", "content": delta.reasoning_content}
            if delta.content:
                yield {"type": "text", "content": delta.content}

class StreamableLLMClient:
    @staticmethod
    async def process_llm_response(llm_response: str) -> str:
        try:
            json_match = re.search(r"\[.*?\]", llm_response, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
                logging.info(f"Extracted JSON content: {json_content}")
            else:
                json_content = llm_response

            tool_calls = json.loads(json_content)
            results = ""
            for tool_call in tool_calls:
                if "tool" in tool_call and "arguments" in tool_call:
                    logging.info(f"Executing tool: {tool_call['tool']}")
                    logging.info(f"With arguments: {tool_call['arguments']}")
                    mcp_config = {
                        "mcpServers": {
                            "test_server": {
                                "command": "url",
                                "url": f"http://localhost:3000/mcp",
                            }
                        }
                    }
                    mcp_servers = {
                        server_name: parse_mcp_client(config)
                        for server_name, config in mcp_config["mcpServers"].items()
                    }
                    for server_name in mcp_servers:
                        async with mcp_servers[server_name] as server:
                            tools = await asyncio.wait_for(
                                server.list_tools(), timeout=60.0
                            )
                            if any(tool.name == tool_call["tool"] for tool in tools):
                                try:
                                    result: list = await server.call_tool(
                                        tool_call["tool"],
                                        (
                                            tool_call["arguments"]
                                            if len(tool_call["arguments"]) > 0
                                            else None
                                        ),
                                    )
                                    logging.info(
                                        f"{tool_call['tool']} execution result: {result}"
                                    )
                                    results += f"{tool_call['tool']} execution result: {[res.text for res in filter(lambda x: True if isinstance(x, TextContent) else False, result)]}\n"
                                except Exception as e:
                                    error_msg = f"Error executing tool: {str(e)}"
                                    logging.error(error_msg)
                                    return error_msg

            if results:
                logging.info(f"Final results: {results}")
                return results
            return "No server found with tools"
        except json.JSONDecodeError:
            logging.info("No valid JSON found in LLM response")
            return llm_response

    @staticmethod
    def format_for_llm(tool: Tool) -> str:
        args_desc = []
        if "properties" in tool.inputSchema:
            for param_name, param_info in tool.inputSchema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in tool.inputSchema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
            Tool Name: {tool.name}
            Description: {tool.description}
            Arguments:
            {chr(10).join(args_desc)}
        """

    @staticmethod
    async def _get_agent_response_streaming(
        messages, llm_url, api_key
    ) -> AsyncGenerator[str, None]:
        
        mcp_config = {
            "mcpServers": {
                "test_server": {
                    "command": "url",
                    "url": f"http://localhost:3000/mcp",
                }
            }
        }

        mcp_servers = {
            server_name: parse_mcp_client(config)
            for server_name, config in mcp_config["mcpServers"].items()
        }
        tools_description = ""

        for server_name in mcp_servers:
            async with mcp_servers[server_name] as server:
                tools = await server.list_tools()
                tools_description += f"Service name: {server_name}\n"
                tools_description += "\n".join(
                    [StreamableLLMClient.format_for_llm(tool) for tool in tools]
                )

        system_message = (
            "You are a helpful assistant  have access to these services and the tools they offer:\n\n"
            # 工具描述prompt
            f"{tools_description}\n"
            "Choose the appropriate tool based on the user's question. "
            "If no tool is needed, reply directly.\n\n"
            "IMPORTANT: When you need to use a tool, you must ONLY respond with "
            "the exact JSON list object format below, nothing else:\n"
            "[{\n"
            '    "tool": "tool-name-1",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "},\n"
            "{\n"
            '    "tool": "tool-name-2",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "},]\n\n"
            "When using the tool, user will return the result, so please be careful to distinguish it.\n"
            # 时间处理prompt
            f"When the user does not provide a specific date, the system uses {datetime.date.today()} as the baseline to coumpute the target date based on the user's intent"
            "The dates/times you provide should must match the user's input exactly, be factually accurate, and must not fabricate false dates."
            "After receiving a tool's response:\n"
            "1. Transform the raw data into a natural, conversational response\n"
            "2. Keep responses concise but informative\n"
            "3. Focus on the most relevant information\n"
            "4. Use appropriate context from the user's question\n"
            "5. Avoid simply repeating the raw data\n\n"
            "Please use only the tools that are explicitly defined above."
        )
        messages = [{"role": "system", "content": system_message}] + messages

        llm_response = ""
        async for chunk in LLMClient.get_stream_response_reasion_and_content(
            messages, llm_url, api_key
        ):
            if chunk["type"] == "text":
                llm_response += chunk["content"]
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": chunk["content"],
            }
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "\n",
        }

        result = await StreamableLLMClient.process_llm_response(llm_response)
        while result != llm_response:
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": result})
            llm_response = ""
            async for chunk in LLMClient.get_stream_response_reasion_and_content(
                messages, llm_url, api_key
            ):
                if chunk["type"] == "text":
                    llm_response += chunk["content"]
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": chunk["content"],
                }
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "\n",
            }
            logging.info(f"\nAssistant: {llm_response}")
            result = await StreamableLLMClient.process_llm_response(llm_response)

        yield {"is_task_complete": True, "require_user_input": False, "content": result}


def parse_mcp_client(config: dict[str, any]):
    command = shutil.which("npx") if config["command"] == "npx" else config["command"]
    if command is None:
        raise ValueError("Command not found")
    return Client(config["url"])