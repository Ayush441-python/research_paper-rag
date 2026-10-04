from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric

from src.rag_pipeline import RagPipeline

load_dotenv()

GOLDEN_PATH = "goldens/correctness_dataset.json"
JUDGE_MODEL = "gpt-4o-mini"
THRESHOLD = 0.7


def load_goldens(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(rag):
    goldens = load_goldens(GOLDEN_PATH)

    test_cases = []

    for golden in goldens:
        result = rag.invoke(golden["question"])

        test_cases.append(
            LLMTestCase(
                input=golden["question"],
                actual_output=result["answer"],
                expected_output=golden["ideal_answer"]
            )
        )

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Compare the factual claims in the actual output against the expected output.",
            "A claim is wrong only if it contradicts the expected output or is factually false.",
            "Judge factual correctness, not completeness or answer length.",
            "Do not penalize brevity or omitted information.",
            "Additional correct information must not lower the score."
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="All stated claims are factually correct and consistent with the expected answer."
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="The answer is mostly correct but contains a minor factual issue."
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="The answer contains a clear factual error or contradiction."
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Identify the key points contained in the expected output.",
            "Check how many of those key points are addressed by the actual output.",
            "Penalize missing or only partially covered key points.",
            "Judge coverage only.",
            "Do not penalize additional information."
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="The answer covers essentially all key points from the expected answer."
            ),
            Rubric(
                score_range=(5, 8),
                expected_outcome="The answer covers the main points but misses one or more important details."
            ),
            Rubric(
                score_range=(0, 4),
                expected_outcome="The answer misses several important points."
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False
    )

    style = GEval(
        name="Style",
        evaluation_steps=[
            "Judge only the teaching style and tone of the actual output.",
            "Reward intuitive and conversational explanations.",
            "Prefer plain language before technical jargon.",
            "Technical terms should be briefly explained when necessary.",
            "Reward prose that feels like a clear teaching explanation.",
            "Do not require an analogy or example.",
            "Penalize stiff, robotic, bureaucratic, or unexplained jargon-heavy writing.",
            "Do not judge correctness or completeness."
        ],
        rubric=[
            Rubric(
                score_range=(9, 10),
                expected_outcome="Clear, intuitive, conversational teaching style with well-explained concepts."
            ),
            Rubric(
                score_range=(7, 8),
                expected_outcome="Clear and conversational with good explanations."
            ),
            Rubric(
                score_range=(4, 6),
                expected_outcome="Understandable but somewhat formal, flat, or list-heavy."
            ),
            Rubric(
                score_range=(0, 3),
                expected_outcome="Dry, robotic, stiff, jargon-heavy, or difficult to follow."
            )
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT
        ],
        threshold=THRESHOLD,
        model=JUDGE_MODEL,
        strict_mode=False
    )

    result = evaluate(
        test_cases=test_cases,
        metrics=[
            correctness,
            completeness,
            style
        ]
    )

    return result


def run_local():
    rag = RagPipeline()
    return run(rag)


if __name__ == "__main__":
    run_local()

