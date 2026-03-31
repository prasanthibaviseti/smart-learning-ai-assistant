from __future__ import annotations

import os
from typing import List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class AnswerGenerator:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.uses_openai = bool(self.api_key and OpenAI is not None)
        self.client = OpenAI(api_key=self.api_key) if self.uses_openai else None
        self.prompt_template = (
            "You are a helpful tutor. Use only the study context below to answer the student.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Give a direct explanation, then one short revision tip."
        )

    def generate(self, question: str, context_items: List[dict]) -> str:
        context = "\n\n".join(
            f"{item['title']} ({item['topic']}): {item['content']}" for item in context_items
        )
        prompt = self.prompt_template.format(context=context, question=question)

        if self.uses_openai and self.client is not None:
            response = self.client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
            )
            return response.output_text.strip()

        return self._fallback_answer(question, context_items)

    def _fallback_answer(self, question: str, context_items: List[dict]) -> str:
        if not context_items:
            return "I could not find relevant study material for that question."

        top = context_items[0]
        supporting = " ".join(item["content"] for item in context_items[1:])
        return (
            f"Based on the retrieved material, the best answer is: {top['content']} "
            f"Additional related context: {supporting} "
            f"Revision tip: summarize this topic in 3 bullet points using the key terms from the answer."
        )
