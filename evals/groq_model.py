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
        for attempt in range(10):
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
                time.sleep(0.3)
                return response.choices[0].message.content

            except groq.RateLimitError as e:
                if attempt == 9:
                    raise e
                wait_time = 4 * (attempt + 1)
                time.sleep(wait_time)
            except Exception as e:
                if attempt == 9:
                    raise e
                time.sleep(2 * (attempt + 1))

    async def a_generate(self, prompt, schema=None):
        return await asyncio.to_thread(
            self.generate,
            prompt,
            schema
        )

    def get_model_name(self):
        return self.model_name