import asyncio
from model_query import LlmManager

manager = LlmManager()

async def query_models(query):
    gemini_response = await(manager.query_gemini(query))
    mistral_response = await(manager.query_mistral(query))
    cohere_response = await(manager.query_cohere(query))
    print("Gemini: ", gemini_response)
    print("Mistral: ", mistral_response)
    print("Cohere: ", cohere_response)

asyncio.run(query_models("Tell me a programming joke."))