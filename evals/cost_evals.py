from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.generator import prompt, llm

load_dotenv()

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 3

PRICE_INPUT_PER_1M = 0.15
PRICE_CACHED_INPUT_PER_1M = 0.075
PRICE_OUTPUT_PER_1M = 0.60

QUERIES_PER_DAY = 2000
USD_TO_INR = 88.0

COST_BUDGET_PER_QUERY_USD = 0.0015

measured_chain = prompt | llm


def measure_tokens(pipeline, question):
    docs = pipeline.retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    response = measured_chain.invoke({
        "question": question,
        "context": context,
    })

    usage = getattr(response, "usage_metadata", {}) or {}

    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0

    input_details = usage.get("input_token_details", {}) or {}

    cached_tokens = (
        input_details.get("cache_read")
        or input_details.get("cached_tokens")
        or 0
    )

    cached_tokens = min(cached_tokens, input_tokens)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
    }


def calculate_cost(input_tokens, output_tokens, cached_tokens):
    uncached_tokens = max(input_tokens - cached_tokens, 0)

    input_cost = (
        uncached_tokens / 1_000_000
    ) * PRICE_INPUT_PER_1M

    cached_cost = (
        cached_tokens / 1_000_000
    ) * PRICE_CACHED_INPUT_PER_1M

    output_cost = (
        output_tokens / 1_000_000
    ) * PRICE_OUTPUT_PER_1M

    total_cost = input_cost + cached_cost + output_cost

    return {
        "input_cost": input_cost,
        "cached_cost": cached_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


def benchmark(pipeline):
    results = []

    for question in QUESTIONS:
        for _ in range(REPEATS):
            tokens = measure_tokens(
                pipeline,
                question,
            )

            costs = calculate_cost(
                tokens["input_tokens"],
                tokens["output_tokens"],
                tokens["cached_tokens"],
            )

            results.append({
                "question": question,
                **tokens,
                **costs,
            })

    return results


def average(results, key):
    return sum(row[key] for row in results) / len(results)


def report(results):
    if not results:
        raise ValueError("No evaluation results were generated.")

    sample_count = len(results)

    avg_input = average(results, "input_tokens")
    avg_output = average(results, "output_tokens")
    avg_cached = average(results, "cached_tokens")

    avg_input_cost = average(results, "input_cost")
    avg_cached_cost = average(results, "cached_cost")
    avg_output_cost = average(results, "output_cost")
    avg_total_cost = average(results, "total_cost")

    min_cost = min(
        row["total_cost"]
        for row in results
    )

    max_cost = max(
        row["total_cost"]
        for row in results
    )

    input_cost = avg_input_cost + avg_cached_cost

    input_percentage = (
        input_cost / avg_total_cost * 100
        if avg_total_cost
        else 0
    )

    output_percentage = (
        avg_output_cost / avg_total_cost * 100
        if avg_total_cost
        else 0
    )

    daily_cost = avg_total_cost * QUERIES_PER_DAY
    monthly_cost = daily_cost * 30

    budget_status = (
        "PASS"
        if avg_total_cost <= COST_BUDGET_PER_QUERY_USD
        else "FAIL"
    )

    print()
    print("=" * 70)
    print("RAG COST EVALUATION")
    print("=" * 70)

    print(f"Model                  : {getattr(llm, 'model_name', 'configured LLM')}")
    print(f"Samples                : {sample_count}")
    print(f"Questions              : {len(QUESTIONS)}")
    print(f"Repeats/question      : {REPEATS}")

    print("-" * 70)

    print(f"Average input tokens   : {avg_input:.0f}")
    print(f"Average cached tokens  : {avg_cached:.0f}")
    print(f"Average output tokens  : {avg_output:.0f}")

    print("-" * 70)

    print(f"Average input cost     : ${input_cost:.6f}")
    print(f"Average output cost    : ${avg_output_cost:.6f}")
    print(f"Average cost/query     : ${avg_total_cost:.6f}")
    print(f"Average cost/query INR : ₹{avg_total_cost * USD_TO_INR:.4f}")

    print("-" * 70)

    print(
        f"Cost range             : "
        f"${min_cost:.6f} - ${max_cost:.6f}"
    )

    print(
        f"Cost distribution      : "
        f"{input_percentage:.1f}% input / "
        f"{output_percentage:.1f}% output"
    )

    print("-" * 70)

    print(f"Traffic                : {QUERIES_PER_DAY:,} queries/day")
    print(f"Estimated daily cost   : ${daily_cost:.2f}")
    print(f"Estimated monthly cost : ${monthly_cost:.2f}")
    print(f"Estimated monthly INR  : ₹{monthly_cost * USD_TO_INR:.2f}")

    print("-" * 70)

    print(
        f"Budget/query           : "
        f"${COST_BUDGET_PER_QUERY_USD:.6f}"
    )

    print(
        f"Budget verdict         : "
        f"{budget_status}"
    )

    print("=" * 70)


def main():
    pipeline = RagPipeline()

    results = benchmark(pipeline)

    report(results)


if __name__ == "__main__":
    main()