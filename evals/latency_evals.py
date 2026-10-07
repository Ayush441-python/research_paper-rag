import math
import time

from dotenv import load_dotenv

from src.rag_pipeline import RagPipeline
from src.llm.generator import generate, generate_stream

load_dotenv()

QUESTIONS = [
    "What is the difference between reference-based and reference-free evals?",
    "Explain what faithfulness measures in a RAG pipeline.",
    "How does the G-Eval metric assign a score?",
    "What is MMLU and why is contamination a problem?",
]

REPEATS = 5
WARMUP_RUNS = 2

SLO_P95_MS = 3000
SLO_TTFT_P95_MS = 1200


def run_end_to_end(pipeline, question):
    result = pipeline.invoke(question)
    return result["answer"]


def run_streaming(pipeline, question):
    retrieval_start = time.perf_counter()

    docs = pipeline.retriever.invoke(question)
    context = [doc.page_content for doc in docs]

    generation_start = time.perf_counter()

    first_token_time = None
    pieces = []

    for piece in generate_stream(question, context):
        if first_token_time is None and piece:
            first_token_time = time.perf_counter()
        pieces.append(piece)

    end_time = time.perf_counter()

    return "".join(pieces), {
        "retrieval": (generation_start - retrieval_start) * 1000,
        "generation": (end_time - generation_start) * 1000,
        "ttft": (
            (first_token_time - retrieval_start) * 1000
            if first_token_time is not None
            else float("nan")
        ),
    }


def percentile(values, p):
    values = [value for value in values if not math.isnan(value)]

    if not values:
        return float("nan")

    values = sorted(values)
    position = (len(values) - 1) * (p / 100)

    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return values[low]

    return (
        values[low] * (high - position)
        + values[high] * (position - low)
    )


def summarize(values):
    values = [value for value in values if not math.isnan(value)]

    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def benchmark(pipeline):
    print(f"Warming up ({WARMUP_RUNS} runs, discarded)...")

    for i in range(WARMUP_RUNS):
        run_end_to_end(pipeline, QUESTIONS[i % len(QUESTIONS)])

    total_ms = []
    retrieval_ms = []
    generation_ms = []
    ttft_ms = []
    answer_lengths = []

    print("Measuring...")

    for question in QUESTIONS:
        for _ in range(REPEATS):
            start = time.perf_counter()

            answer, metrics = run_streaming(pipeline, question)

            elapsed = (time.perf_counter() - start) * 1000

            total_ms.append(elapsed)
            retrieval_ms.append(metrics["retrieval"])
            generation_ms.append(metrics["generation"])
            ttft_ms.append(metrics["ttft"])
            answer_lengths.append(len(answer or ""))

    return {
        "total": total_ms,
        "retrieval": retrieval_ms,
        "generation": generation_ms,
        "ttft": ttft_ms,
        "answer_len": answer_lengths,
    }


def print_row(label, values):
    stats = summarize(values)

    print(
        f"{label:<12} | "
        f"n={stats['n']:<3} "
        f"mean={stats['mean']:7.1f}  "
        f"p50={stats['p50']:7.1f}  "
        f"p95={stats['p95']:7.1f}  "
        f"p99={stats['p99']:7.1f}  "
        f"min={stats['min']:7.1f}  "
        f"max={stats['max']:7.1f}"
    )


def print_slo(label, p95, budget):
    verdict = "PASS" if p95 <= budget else "FAIL"

    print(
        f"SLO: {label:<22} "
        f"p95 <= {budget:>5} ms  "
        f"->  p95 = {p95:7.0f} ms   "
        f"[{verdict}]"
    )


def report(results):
    total = summarize(results["total"])
    ttft = summarize(results["ttft"])

    print("\n" + "=" * 78)
    print("LATENCY (milliseconds)")
    print("=" * 78)

    print(
        f"{'stage':<12} | "
        f"{'samples':<7}"
        f"{'mean':>11}"
        f"{'p50':>11}"
        f"{'p95':>11}"
        f"{'p99':>11}"
        f"{'min':>11}"
        f"{'max':>11}"
    )

    print("-" * 78)

    print_row("end-to-end", results["total"])
    print_row("ttft", results["ttft"])
    print_row("retrieval", results["retrieval"])
    print_row("generation", results["generation"])

    average_length = sum(results["answer_len"]) / len(results["answer_len"])

    print("-" * 78)
    print(f"avg answer length: {average_length:.0f} chars")

    print("=" * 78)

    print_slo("full answer", total["p95"], SLO_P95_MS)
    print_slo("first token (perceived)", ttft["p95"], SLO_TTFT_P95_MS)

    print("=" * 78)


def main():
    pipeline = RagPipeline()
    results = benchmark(pipeline)
    report(results)


if __name__ == "__main__":
    main()












































"""
Operational eval: LATENCY (with time-to-first-token).

Unlike every eval we built before this (correctness, faithfulness, toxicity,
leakage, scope), latency needs no golden dataset and no LLM judge. It is a
deterministic measurement: run the pipeline N times, collect a distribution,
and report percentiles against a budget (SLO) -- not against a ground truth.

Two latency numbers matter, and they answer different questions:
  - END-TO-END total : how long until the FULL answer is ready
  - TTFT (perceived) : how long until the user sees the FIRST token stream in
For a streaming chat UI, TTFT is what "feels fast" -- the total can be long
while the experience is snappy. We measure both.

Key ideas encoded below:
  - perf_counter, not time()          (right clock for elapsed time)
  - many samples -> percentiles       (p95/p99 tail, not the misleading mean)
  - discard warmup                    (cold start poisons the stats)
  - decompose the pipeline            (retrieval + generation, + TTFT)
  - log answer length                 (latency couples to output length)
  - single-user only                  (load testing is a separate exercise)
"""
