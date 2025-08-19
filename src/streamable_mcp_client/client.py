import asyncio
import json
import logging
import os
import re
import shutil
from typing import AsyncGenerator

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport, SSETransport
from mcp.types import TextContent, Tool
import datetime

from openai import AsyncOpenAI

class LLMClient:
    @staticmethod
    async def get_stream_response_reasion_and_content(
        messages: list[dict[str, str]], model_name: str, llm_url: str, api_key: str
    ) -> AsyncGenerator[str, None]:
        
        client: AsyncOpenAI = AsyncOpenAI(api_key=api_key, base_url=llm_url)

        response = await client.chat.completions.create(
            messages=messages, stream=True, model=model_name
        )
        async for chunk in response:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content'):
                if delta.reasoning_content:
                    yield {"type": "thought", "content": delta.reasoning_content}
            if delta.content:
                yield {"type": "text", "content": delta.content}

class StreamableLLMClient:
    @staticmethod
    async def process_llm_response(mcp_config, llm_response: str) -> str:
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

                    mcp_servers = {
                        server_name: parse_mcp_client(config)
                        for server_name, config in mcp_config["mcpServers"].items()
                    }
                    for server_name in mcp_servers:
                        async with mcp_servers[server_name] as server:
                            print(f"Connecting to server: {server_name}")
                            tools = await asyncio.wait_for(
                                server.list_tools(), timeout=60.0
                            )
                            if any(tool.name == tool_call["tool"] for tool in tools):
                                
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
                                text_contents = extract_text_contents(result)
                                results += f"{tool_call['tool']} execution result: {text_contents}\n"
                                

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
        mcp_config, messages, model_name, llm_url, api_key
    ) -> AsyncGenerator[str, None]:

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
            f"When the user does not provide a specific date, the system uses {datetime.date.today()} as the baseline to coumpute the target date based on the user's intent"
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

        result = "start_process_llm_response"
        llm_response = "start_llm_response"
        while result != llm_response:
            llm_response = ""
            async for chunk in LLMClient.get_stream_response_reasion_and_content(
                messages, model_name, llm_url, api_key
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
            result = await StreamableLLMClient.process_llm_response(mcp_config, llm_response)
            
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "user", "content": result})

        yield {"is_task_complete": True, "require_user_input": False, "content": result}

def extract_text_contents(obj):
    text_contents = []
    
    # 如果是 TextContent，直接提取 text
    if isinstance(obj, TextContent):
        text_contents.append(obj.text)
    
    # 如果是可迭代对象（如 list、tuple），递归遍历
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            text_contents.extend(extract_text_contents(item))
    
    # 如果是字典或类似对象（如 dataclass、namedtuple），递归遍历值
    elif hasattr(obj, '__dict__'):  # 普通对象
        for value in obj.__dict__.values():
            text_contents.extend(extract_text_contents(value))
    elif isinstance(obj, dict):  # 字典
        for value in obj.values():
            text_contents.extend(extract_text_contents(value))
    
    return text_contents

def parse_mcp_client(config: dict[str, any]):
    
    command = config["command"]
    if command is None:
        raise ValueError("Command not found")
    
    match command:
        case "streamablehttp":
            return Client(
                StreamableHttpTransport(
                    url=config["url"],
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Accept": "application/json", 
                        "Accept-Language": "en-US,en;q=0.9",  
                        "Accept-Encoding": "gzip, deflate, br",  
                        "Connection": "keep-alive",  
                        **({"Authorization": f"Bearer {config['api_key']}"} if config.get("api_key") else {}),
                        "Cache-Control": "no-cache", 
                    },
                    sse_read_timeout=config["sse_read_timeout"] if config.get("sse_read_timeout") else 60,
                )
            )
            
        case "sse":
            return Client(
                SSETransport(
                    url=config["url"],
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Accept": "application/json", 
                        "Accept-Language": "en-US,en;q=0.9",  
                        "Accept-Encoding": "gzip, deflate, br",  
                        "Connection": "keep-alive",  
                        **({"Authorization": f"Bearer {config['api_key']}"} if config.get("api_key") else {}),
                        "Cache-Control": "no-cache", 
                    },
                    sse_read_timeout=config["sse_read_timeout"] if config.get("sse_read_timeout") else 60,
                )
            )
        
        case "npx" | "uv" | "uvx":
            resolved = shutil.which(command)
            return Client(
                StdioTransport(
                    command=resolved,
                    args=config["args"],
                    env={**os.environ, **config["env"]} if config.get("env") else None,
                )
            )