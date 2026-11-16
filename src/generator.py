from typing import Dict, List, Optional
import os
from openai import OpenAI

# Groq API configuration defined directly in code
# (No .env file required for model selection or endpoint)
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
FALLBACK_GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TEMPERATURE = 0.2

# Strict system grounding instructions matching the notebook
STRICT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based ONLY on the provided context. "
    "If the context does not contain information to answer this question, "
    "say 'I don't have enough context or information to answer this question'. "
    "Do not make up information or assume anything. Strictly answer only from the provided context."
)


class RAGGenerator:
    """
    Handles LLM response generation using Groq via the OpenAI client interface.
    The Groq model ('openai/gpt-oss-20b') is defined directly in code,
    with an automatic fallback to 'llama-3.3-70b-versatile' if required.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = GROQ_BASE_URL,
        model: str = DEFAULT_GROQ_MODEL,
        fallback_model: str = FALLBACK_GROQ_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.fallback_model = fallback_model
        self.temperature = temperature
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GROQ_API_KEY is not set. Please provide it in the UI sidebar or set it in your environment."
                )
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def build_prompt_messages(self, question: str, chunks: List[str]) -> List[Dict[str, str]]:
        """Construct the strict system grounding prompt and user query."""
        context = "\n\n".join(chunks)
        return [
            {
                "role": "system",
                "content": STRICT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext: {context}",
            },
        ]

    def generate(
        self,
        question: str,
        chunks: List[str],
        metadatas: Optional[List[Dict[str, any]]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate answer from retrieved context using the Groq backend model.
        Catches deprecated/unavailable model errors and automatically retries with
        the fallback model.
        """
        if not chunks:
            return "I don't have enough context or information to answer this question."

        messages = self.build_prompt_messages(question, chunks)
        primary_model = model or self.model

        try:
            response = self.client.chat.completions.create(
                model=primary_model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            # If the primary model identifier is retired or unavailable, fall back seamlessly
            if "model" in err_str and ("not found" in err_str or "does not exist" in err_str or "decommissioned" in err_str):
                try:
                    fallback = self.fallback_model
                    response = self.client.chat.completions.create(
                        model=fallback,
                        messages=messages,
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content
                except Exception as inner_e:
                    raise RuntimeError(f"Generation failed with fallback model '{fallback}': {inner_e}") from inner_e
            raise RuntimeError(f"Generation failed with model '{primary_model}': {e}") from e
