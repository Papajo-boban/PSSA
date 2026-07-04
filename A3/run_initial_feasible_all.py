import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from pathlib import Path
from typing import Dict, List

from a3_initial_solution import initial_feasible_solution
from pms_instance import PMSInstance


def _run_instance(task: Dict) -> Dict:
    instance_path = Path(task["instance_path"])
    output_dir = Path(task["output_dir"])
    frontier_size = task["frontier_size"]
    rollback_size = task["rollback_size"]
    max_rollbacks = task["max_rollbacks"]
    time_limit_small = task["time_limit_small"]
    time_limit_large = task["time_limit_large"]
    disable_a2_fallback = task["disable_a2_fallback"]

    started_at = time.time()
    try:
        instance = PMSInstance(str(instance_path))
        is_large_instance = instance.n_jobs >= 500
        raw_time_limit = time_limit_large if is_large_instance else time_limit_small
        total_time_limit_s = None if raw_time_limit <= 0 else raw_time_limit
        result = initial_feasible_solution(
            instance,
            frontier_size=frontier_size,
            rollback_size=rollback_size,
            max_rollbacks=max_rollbacks,
            total_time_limit_s=total_time_limit_s,
            allow_a2_fallback=not disable_a2_fallback,
            fast_large_instance_mode=is_large_instance,
            diagnostics_enabled=True,
        )
        output_path = output_dir / f"{instance_path.stem}.solution.json"
        output_path.write_text(json.dumps(result["solution"], indent=2), encoding="utf-8")
        runtime = round(time.time() - started_at, 2)
        budget_label = "none" if total_time_limit_s is None else f"{total_time_limit_s}s"
        return {
            "status": "ok",
            "instance_name": instance_path.name,
            "runtime": runtime,
            "budget_label": budget_label,
            "objective": result["objective"],
            "tardiness": result["tardiness"],
            "makespan": result["makespan"],
        }
    except Exception as exc:
        runtime = round(time.time() - started_at, 2)
        return {
            "status": "fail",
            "instance_name": instance_path.name,
            "runtime": runtime,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _build_task(instance_path: Path, args: argparse.Namespace) -> Dict:
    return {
        "instance_path": str(instance_path),
        "output_dir": str(args.output_dir),
        "frontier_size": args.frontier_size,
        "rollback_size": args.rollback_size,
        "max_rollbacks": args.max_rollbacks,
        "time_limit_small": args.time_limit_small,
        "time_limit_large": args.time_limit_large,
        "disable_a2_fallback": args.disable_a2_fallback,
    }


def _print_result(result: Dict) -> None:
    if result["status"] == "ok":
        print(
            f"[OK] {result['instance_name']} | runtime={result['runtime']}s budget={result['budget_label']} "
            f"objective={result['objective']} tardiness={result['tardiness']} makespan={result['makespan']}"
        )
        return

    print(
        f"[FAIL] {result['instance_name']} | runtime={result['runtime']}s "
        f"{result['error_type']}: {result['error']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the A3 initial feasible solution builder on all instances.")
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=Path("instances"),
        help="Directory containing instance JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("solutions"),
        help="Directory where solution JSON files will be written.",
    )
    parser.add_argument("--frontier-size", type=int, default=12)
    parser.add_argument("--rollback-size", type=int, default=8)
    parser.add_argument("--max-rollbacks", type=int, default=200)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of instances to solve in parallel. Use 1 for serial execution.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--time-limit-small",
        type=int,
        default=0,
        help="Per-instance time limit in seconds for non-j500 instances. Use 0 to disable.",
    )
    parser.add_argument(
        "--time-limit-large",
        type=int,
        default=0,
        help="Per-instance time limit in seconds for j500 instances. Use 0 to disable.",
    )
    parser.add_argument(
        "--disable-a2-fallback",
        action="store_true",
        help="Disable the heavy A2 fallback constructor.",
    )
    args = parser.parse_args()

    instance_paths = sorted(args.instances_dir.glob("*.json"))
    if args.limit is not None:
        instance_paths = instance_paths[: args.limit]

    if not instance_paths:
        print(f"No instance files found in {args.instances_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    failures = 0
    tasks: List[Dict] = []
    for instance_path in instance_paths:
        print(f"[RUN] {instance_path.name}")
        tasks.append(_build_task(instance_path, args))

    if args.workers <= 1:
        for task in tasks:
            result = _run_instance(task)
            _print_result(result)
            if result["status"] == "ok":
                successes += 1
            else:
                failures += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_instance, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                _print_result(result)
                if result["status"] == "ok":
                    successes += 1
                else:
                    failures += 1

    print()
    print(f"Summary: successes={successes}, failures={failures}, total={len(instance_paths)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
