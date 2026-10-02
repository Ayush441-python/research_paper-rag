import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

from evals.groq_model import GroqModel
from src.retriever.eval_pipeline import create_eval_retriever

load_dotenv()

GOLDEN_PATH = "goldens/retriever_dataset.json"
THRESHOLD = 0.7

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

retriever = create_eval_retriever(k=5)

model_name = os.getenv("EVALS_MODEL", "openai/gpt-oss-120b")
model = GroqModel(model_name)

test_cases = []

for g in dataset["data"]:
    question = g["question"]
    ideal_answer = g["ideal_answer"]

    retrieved_docs = retriever.invoke(
        question
    )

    retrieval_context = [
        doc.page_content
        for doc in retrieved_docs
    ]

    test_case = LLMTestCase(
        input=question,
        actual_output=ideal_answer,
        expected_output=ideal_answer,
        retrieval_context=retrieval_context
    )

    test_cases.append(test_case)


relevancy_metric = ContextualRelevancyMetric(
    threshold=THRESHOLD,
    model=model,
    async_mode=True
)

precision_metric = ContextualPrecisionMetric(
    threshold=THRESHOLD,
    model=model,
    async_mode=False
)

recall_metric = ContextualRecallMetric(
    threshold=THRESHOLD,
    model=model,
    async_mode=False
)

from deepeval.evaluate.configs import AsyncConfig

evaluate(
    test_cases=test_cases,
    metrics=[relevancy_metric, precision_metric, recall_metric],
    async_config=AsyncConfig(run_async=False)
)
