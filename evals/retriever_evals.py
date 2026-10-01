import os
import json

from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from evals.groq_model import GroqModel
from src.retriever.pipeline import create_retriever

load_dotenv()

GOLDEN_PATH = "goldens/retriever_dataset.json"
THRESHOLD = 0.7

with open(GOLDEN_PATH,"r",encoding="utf-8") as f:
    dataset = json.load(f)

retriever = create_retriever(k=5)

model = GroqModel(os.getenv("EVALS_MODEL"))

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


metric = AnswerRelevancyMetric(
    threshold=THRESHOLD,
    model = model
)

evaluate(
    test_cases=test_cases,
    metrics=[metric]
)