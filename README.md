# streamable-mcp-client
Let the MCP client see their thought process as they invoke the tool, instead of waiting for the tool to return a result

## 🛸 Usage Description
You just send mcp_servers_config, messages, llm model name, llm base url and llm api key to `StreamableLLMClient._get_agent_response_streaming()`, you will get a streaming thought and response from the mcp server.

## 💡 Example Usage for DeepSeek
```python
import asyncio
from src.streamable_mcp_client.client import StreamableLLMClient

async def main():
    mcp_servers_config = {
        "mcpServers": {
            "test_server": {
                "command": "streamablehttp",
                "url": "http://localhost:3000/mcp" # your mcp url
            }
        }
    } # your mcp config

    messages = [{"role": "user", "content": "Help me to call the test_server"}]
    llm_model_name = "deepseek-chat" # your model name
    llm_url = "https://api.deepseek.com/v1" # your llm url
    llm_api_key = "your_api_key" # your llm api key

    try:
        async for result in StreamableLLMClient._get_agent_response_streaming(
            mcp_config=mcp_servers_config, 
            messages=messages,
            model_name=llm_model_name, 
            llm_url=llm_url, 
            api_key=llm_api_key
        ):
            print(result["content"], end="")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
# result: {"is_task_complete": bool, "require_user_input": bool, "content": "streaming response"}

```

## 💡 Input Parameters
- `mcp_servers_config`: A `dict` of MCP servers configuration.
   - `command`: `streamablehttp`, `sse` and `npx | uv | uvx`
   - `url`: The URL of the MCP server.
   - `api_key`: The api_key for the MCP server.
   - `timeout`: The timeout for the MCP server.
   - `args`: if you not want to use streamablehttp and sse, you don't need to set url and api_key, instead, you can use args to set the command and args. About args standard, you can refer to [official mcp servers](https://github.com/modelcontextprotocol/servers)
- `messages`: A `list[dict]` of messages to be sent to the LLM.
- `model_name`: The name of the LLM model to be used.
- `base_url`: The base URL of the LLM API.
- `api_key`: The API key to be used to authenticate with the LLM API.

### Example for streamablehttp and sse:
```python
# mcp_servers_config for streamablehttp
mcp_servers_config = {
  "mcpServers": {
    "test_server": {
      "command": "streamablehttp",
      "url": "http://localhost:3000/mcp"
    }
  }
}

# mcp_servers_config for sse
mcp_servers_config = {
  "mcpServers": {
    "test_server": {
      "command": "sse",
      "url": "http://localhost:3000/mcp"
    }
  }
}
```

### Example for stdio:
```python
# This is a official mcp server config
mcp_servers_config = {
  "mcpServers": {
    "everything": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-everything"
      ]
    }
  }
}
```

## 💡 Output Parameters
- `is_task_complete`: True/False, indicates whether the task is completed or not, if True, the content is the final result of mcp server call completion, if False, the result is streaming response of llm thought how to call mcp server.

- `require_user_input`: True/False, indicates whether the LLM requires user input or not.

- `content`: streaming response