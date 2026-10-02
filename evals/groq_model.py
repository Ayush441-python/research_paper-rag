import asyncio
import time
import groq
from deepeval.models import DeepEvalBaseLLM


class GroqModel(DeepEvalBaseLLM):

    def __init__(self, model):
        self.model_name = model
        self.client = groq.Groq()

    def load_model(self):
        return self.client

    def generate(self, prompt, schema=None):
        for attempt in range(5):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0
                }

                if schema is not None:
                    kwargs["response_format"] = {
                        "type": "json_object"
                    }

                response = self.client.chat.completions.create(**kwargs)

                return response.choices[0].message.content

            except groq.RateLimitError as e:
                if attempt == 4:
                    raise e

                time.sleep(5 * (attempt + 1))

    async def a_generate(self, prompt, schema=None):
        return await asyncio.to_thread(
            self.generate,
            prompt,
            schema
        )

    def get_model_name(self):
        return self.model_name