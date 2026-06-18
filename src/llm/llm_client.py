# This file handles talking to the Groq API.
# Groq gives us free, very fast access to LLaMA 3.3 70B.

import os
from groq import Groq
from dotenv import load_dotenv

# Load the GROQ_API_KEY from .env file
load_dotenv()


class LLMClient:
    def __init__(self, model: str = "llama-3.3-70b-versatile", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        
        # Initialize Groq client (reads GROQ_API_KEY from environment automatically)
        self.client = Groq()

    def get_answer(self, prompt: str) -> str:
        """
        Sends the prompt to Groq and returns the LLM's answer.
        
        Args:
            prompt: the full prompt string (with context + question)
        
        Returns:
            The LLM's response as a string
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,          # low temperature = more factual, less creative
            max_completion_tokens=self.max_tokens,
            top_p=1,
            stream=False,             # we get the full response at once (simpler)
            stop=None,
        )
        
        answer = completion.choices[0].message.content
        return answer
