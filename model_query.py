import os
import asyncio
import cohere
import google.generativeai as genai
from mistralai.client import MistralClient

class LlmManager():
    def __init__(self):
        api_key_cohere = 'w1FP9IRl1UevS6LO5A2jRCIMO0KvYiPQNvmm5Z3x'
        api_key_gemini = 'AIzaSyC6AWqSlK1JWRIe79kwR48-WOr9WKplzy8'
        api_key_mistral = 'EnR9bolLjxLoo1eVHoyY1lv2ztSjLWFO'

        self.model = "mistral-medium"
        self.mistral_client = MistralClient(api_key=api_key_mistral)
        genai.configure(api_key=api_key_gemini)
        self.gemini = genai.GenerativeModel("gemini-1.5-flash")
        self.co = cohere.ClientV2(api_key_cohere)

    async def query_gemini(self, query):
        return self.gemini.generate_content(query)

    async def query_mistral(self, query):
        messages = [{"role": "user", "content": query}]
        response = self.mistral_client.chat(model=self.model, messages=messages)
        return response

    async def query_cohere(self, query):
        return self.co.chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": query}]
        )
