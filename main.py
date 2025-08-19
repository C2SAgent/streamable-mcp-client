import asyncio
from src.streamable_mcp_client.client import StreamableLLMClient

# example
async def main():
    mcp_servers_config = {
        "mcpServers": {
            "test_server": {
                "command": "streamablehttp",
                "url": "http://localhost:3001/mcp" # your mcp url
            }
        }
    } # your mcp config

    messages = [{"role": "user", "content": "帮我查询一下今天的万年历"}]
    llm_model_name = "deepseek-chat" # your model name
    llm_url = "https://api.deepseek.com/v1" # your llm url
    llm_api_key = "YOUR_API_KEY" # your llm api key

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