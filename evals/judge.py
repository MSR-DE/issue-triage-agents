"""DeepEval judge on Groq (gpt-oss-120b) instead of DeepEval's default OpenAI model,
which needs a paid key. DeepEval calls generate(prompt, schema); with a schema we
return a parsed instance of it (JSON mode), otherwise plain text."""
from deepeval.models import DeepEvalBaseLLM
from langchain_groq import ChatGroq


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model="openai/gpt-oss-120b"):
        self.model_name = model
        self.llm = ChatGroq(model=model, temperature=0, reasoning_effort="low",
                            max_tokens=2048, max_retries=5)

    def load_model(self):
        return self.llm

    def generate(self, prompt, schema=None):
        if schema is None:
            return self.llm.invoke(prompt).content
        return self.llm.with_structured_output(schema, method="json_mode").invoke(prompt)

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"groq/{self.model_name}"
