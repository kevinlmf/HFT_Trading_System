"""Statistics Computing Benchmark

Compare end-to-end performance and outcomes with/without the optimization stack.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from Execution.engine.pipeline import create_sample_data
from Optimization.integrated_optimization import IntegratedOptimizedFlow


@dataclass
class StrategySummary:
    strategy: str
    score: float
    sharpe: float
    total_return: float
    passed_risk: bool


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # pylint: disable=broad-except
        return 0.0


def _extract_top_strategies(results: List[Any], top_k: int = 3) -> List[StrategySummary]:
    summaries: List[StrategySummary] = []
    for res in results[:top_k]:
        total_return = getattr(res, "total_return", getattr(res, "cumulative_return", 0.0))
        summaries.append(
            StrategySummary(
                strategy=getattr(res, "strategy_name", "unknown"),
                score=_safe_float(getattr(res, "overall_score", 0.0)),
                sharpe=_safe_float(getattr(res, "sharpe_ratio", 0.0)),
                total_return=_safe_float(total_return),
                passed_risk=bool(getattr(res, "passed_risk_checks", False)),
            )
        )
    return summaries


def run_flow(use_stack: bool, data: pd.DataFrame, monte_carlo_paths: int) -> Dict[str, Any]:
    flow = IntegratedOptimizedFlow(
        monte_carlo_paths=monte_carlo_paths,
        use_optimization_stack=use_stack,
    )

    start = time.perf_counter()
    result = flow.execute_complete_flow_optimized(data=data, use_gpu_for_risk=False)
    runtime = time.perf_counter() - start

    final_report = result["final_report"]
    top_strategies = _extract_top_strategies(result["comparison_results"], top_k=5)

    summary = {
        "use_optimization_stack": use_stack,
        "runtime_seconds": runtime,
        "execution_method": final_report.get("execution_method"),
        "data_quality": final_report.get("data_quality"),
        "n_records": final_report.get("data_size"),
        "strategies_tested": final_report.get("n_strategies_tested"),
        "strategies_passed_risk": final_report.get("strategies_passed_risk"),
        "best_strategy": final_report.get("best_strategy"),
        "best_score": final_report.get("best_score"),
        "best_sharpe": final_report.get("best_sharpe"),
        "top_strategies": [asdict(s) for s in top_strategies],
        "optimization_info": result.get("optimization_info", {}),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare optimization stack on/off")
    parser.add_argument(
        "--records",
        type=int,
        default=1000,
        help="Number of synthetic records to generate (default: 1000)",
    )
    parser.add_argument(
        "--monte-carlo-paths",
        type=int,
        default=10000,
        help="Monte Carlo paths for each run (default: 10000)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Statistics Computing Benchmark")
    print("=" * 80)

    # Generate synthetic dataset once for both runs
    data = create_sample_data(n_records=args.records)
    data["close"] = data["price"]

    # Baseline: optimization stack disabled
    print("\nRunning baseline (optimization stack OFF)...")
    baseline_summary = run_flow(use_stack=False, data=data, monte_carlo_paths=args.monte_carlo_paths)

    # Optimized: optimization stack enabled
    print("\nRunning optimized mode (optimization stack ON)...")
    optimized_summary = run_flow(use_stack=True, data=data, monte_carlo_paths=args.monte_carlo_paths)

    # Derive comparison metrics
    speedup = None
    if optimized_summary["runtime_seconds"] > 0:
        speedup = baseline_summary["runtime_seconds"] / optimized_summary["runtime_seconds"]

    comparison = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": {
            "records": args.records,
            "source": "create_sample_data",
        },
        "baseline": baseline_summary,
        "optimized": optimized_summary,
        "runtime_speedup": speedup,
    }

    print("\n" + "-" * 80)
    print("Summary")
    print("-" * 80)
    print(f"Records: {args.records:,} | Monte Carlo paths: {args.monte_carlo_paths:,}")
    print(f"Baseline runtime:   {baseline_summary['runtime_seconds']:.4f} sec")
    print(f"Optimized runtime:  {optimized_summary['runtime_seconds']:.4f} sec")
    if speedup:
        print(f"Runtime speedup:   {speedup:.2f}x")
    else:
        print("Runtime speedup:   N/A")
    print(f"Baseline best strategy:  {baseline_summary['best_strategy']} (score {baseline_summary['best_score']:.2f})")
    print(f"Optimized best strategy: {optimized_summary['best_strategy']} (score {optimized_summary['best_score']:.2f})")

    # Persist results
    out_dir = Path("results/strategies/statistics_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = comparison["timestamp"]

    json_path = out_dir / f"statistics_computing_comparison_{ts}.json"
    text_path = out_dir / f"statistics_computing_comparison_{ts}.txt"

    json_path.write_text(json.dumps(comparison, indent=2))

    text_lines = [
        f"Statistics Computing Benchmark | {ts}",
        "",
        f"Records: {args.records:,} | Monte Carlo paths: {args.monte_carlo_paths:,}",
        "",
        f"Baseline runtime:   {baseline_summary['runtime_seconds']:.4f} sec",
        f"Optimized runtime:  {optimized_summary['runtime_seconds']:.4f} sec",
        f"Runtime speedup:   {speedup:.2f}x" if speedup else "Runtime speedup:   N/A",
        "",
        f"Baseline best strategy:  {baseline_summary['best_strategy']} (score {baseline_summary['best_score']:.2f}, Sharpe {baseline_summary['best_sharpe']:.3f})",
        f"Optimized best strategy: {optimized_summary['best_strategy']} (score {optimized_summary['best_score']:.2f}, Sharpe {optimized_summary['best_sharpe']:.3f})",
        "",
        "Top strategies (baseline):",
    ]

    for entry in baseline_summary["top_strategies"][:3]:
        text_lines.append(
            f"  - {entry['strategy']}: score {entry['score']:.2f}, Sharpe {entry['sharpe']:.3f}, risk_ok={entry['passed_risk']}"
        )

    text_lines.append("")
    text_lines.append("Top strategies (optimized):")
    for entry in optimized_summary["top_strategies"][:3]:
        text_lines.append(
            f"  - {entry['strategy']}: score {entry['score']:.2f}, Sharpe {entry['sharpe']:.3f}, risk_ok={entry['passed_risk']}"
        )

    text_lines.append("")
    text_lines.append("Optimization info:")
    text_lines.append(f"  {optimized_summary['optimization_info']}")

    text_path.write_text("\n".join(text_lines))

    print("")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved text summary to {text_path}")


if __name__ == "__main__":
    main()
