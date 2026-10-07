import json
import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
from evals.groq_model import GroqModel

load_dotenv()

JUDGE_MODEL = GroqModel(os.getenv("EVALS_MODEL", "openai/gpt-oss-120b"))
GOLDEN_PATH = "goldens/faithfullness_dataset.json"
THRESHOLD = 0.7


def load_goldens(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run():
    goldens = load_goldens(GOLDEN_PATH)
    rag = RagPipeline()

    test_cases = []

    for g in goldens:
        result = rag.invoke(g["query"])

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],
                retrieval_context=result["context"],
            )
        )

    metrics = [
        ContextualRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=JUDGE_MODEL,
            include_reason=True,
        ),
    ]

    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    return results


if __name__ == "__main__":
    run()