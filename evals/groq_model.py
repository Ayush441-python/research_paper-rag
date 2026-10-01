from groq import Groq
from deepeval.models import DeepEvalBaseLLM


class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model_name = model
        self.client = Groq()

    def load_model(self):
        return self.client

    def generate(self, prompt, schema=None):

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    async def a_generate(self, prompt, schema=None):

        return self.generate(prompt, schema)

    def get_model_name(self):

        return self.model_name