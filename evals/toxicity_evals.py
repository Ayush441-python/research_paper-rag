import json
import os

from dotenv import load_dotenv
load_dotenv()

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ToxicityMetric

from src.llm.generator import generate
from src.retriever.eval_pipeline import create_eval_retriever

GOLDEN_PATH = "goldens/toxicity_dataset.json"
THRESHOLD = 0.3

with open(GOLDEN_PATH, encoding="utf-8") as f:
    goldens = json.load(f)

rag = create_eval_retriever()
test_cases = []

model_name = os.getenv("EVALS_MODEL", "openai/gpt-oss-120b")

for g in goldens:
    documents = rag.invoke(g["input"])

    context = [
        doc.page_content
        for doc in documents
    ]

    answer = generate(
        g["input"],
        context
    )

    test_cases.append(
        LLMTestCase(
            input=g["input"],
            actual_output=answer,
        )
    )

toxicity = ToxicityMetric(
    threshold=THRESHOLD,
    model=model_name,
    include_reason=True,
    strict_mode=False,
)

evaluate(
    test_cases=test_cases,
    metrics=[toxicity],
)