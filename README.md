# streamable-mcp-client
Let the MCP client see their thought process as they invoke the tool, instead of waiting for the tool to return a result

## Usage Description
You just send mcp_servers_config, messages, llm model name, llm base url and llm api key to `StreamableLLMClient._get_agent_response_streaming()`, you will get a streaming thought and response from the mcp server.

## Example Usage for DeepSeek
```python
mcp_servers_config = {
  "mcpServers": {
    "test_server": {
      "command": "streamablehttp",
      "url": "http://localhost:3000/mcp"
    }
  }
}

messages = [{"role": "user", "content": "how are you?"}]
llm_model_name = "deepseek-chat"
llm_base_url = "https://api.deepseek.com/v1"
llm_api_key = "YOUR_API_KEY"

async for result in StreamableLLMClient._get_agent_response_streaming(
    mcp_config=mcp_servers_config, 
    messages=messages,
    model_name=llm_model_name, 
    base_url=llm_base_url, 
    api_key=llm_api_key
):
    print(result)

# result: {"is_task_complete": bool, "require_user_input": bool, "content": "streaming response"}

```

## Result Analysis
- is_task_complete: True/False, indicates whether the task is completed or not, if True, the content is the final result of mcp server call completion, if False, the result is streaming response of llm thought how to call mcp server.

- require_user_input: True/False, indicates whether the LLM requires user input or not.

- content: streaming response