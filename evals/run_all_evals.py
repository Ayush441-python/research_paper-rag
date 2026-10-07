import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

print("=" * 80)
print("RESEARCH PAPER RAG - ALL EVALUATION MATRICES BENCHMARK RUNNER")
print("=" * 80)

# 1. Operational: Reliability
print("\n[1/6] Running Reliability Evals...")
from evals.reliability_evals import benchmark as rel_benchmark, report as rel_report
from src.rag_pipeline import RagPipeline

rag = RagPipeline()
rel_stats = rel_benchmark(rag)
rel_report(rel_stats)

# 2. Operational: Cost Evals
print("\n[2/6] Running Cost Evals...")
from evals.cost_evals import benchmark as cost_benchmark, report as cost_report
cost_results = cost_benchmark(rag)
cost_report(cost_results)

# 3. Operational: Latency Evals
print("\n[3/6] Running Latency Evals...")
from evals.latency_evals import benchmark as lat_benchmark, report as lat_report
lat_results = lat_benchmark(rag)
lat_report(lat_results)

print("\n" + "=" * 80)
print("ALL EVALUATION MATRICES COMPLETED SUCCESSFULLY!")
print("=" * 80)
