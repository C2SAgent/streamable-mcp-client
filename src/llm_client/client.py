from openai import AsyncOpenAI
from typing import AsyncGenerator


class LLMClient:
    async def get_stream_response_reasion_and_content(
        self, messages: list[dict[str, str]], llm_url, api_key
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
