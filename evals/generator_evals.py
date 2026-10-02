import json

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
from src.llm.generator import generate
from evals.groq_model import GroqModel

load_dotenv()

GOLDEN_PATH = "goldens/faithfullness_dataset.json"
THRESHOLD = 0.7

model = GroqModel("openai/gpt-oss-120b")


def load_dataset():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def run():
    dataset = load_dataset()
    test_cases = []

    for item in dataset:
        question = item["query"]
        context = item["context"]

        answer = generate(question, context)

        test_cases.append(
            LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=context
            )
        )

    metrics = [
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=model,
            include_reason=True
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=model,
            include_reason=True
        )
    ]

    evaluate(
        test_cases=test_cases,
        metrics=metrics
    )

if __name__ == "__main__":
    run()