import argparse
import json
import time
from pathlib import Path

from a3_initial_solution import initial_feasible_solution
from pms_instance import PMSInstance


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

    for instance_path in instance_paths:
        print(f"[RUN] {instance_path.name}")
        started_at = time.time()
        try:
            instance = PMSInstance(str(instance_path))
            is_large_instance = instance.n_jobs >= 500
            raw_time_limit = args.time_limit_large if is_large_instance else args.time_limit_small
            total_time_limit_s = None if raw_time_limit <= 0 else raw_time_limit
            result = initial_feasible_solution(
                instance,
                frontier_size=args.frontier_size,
                rollback_size=args.rollback_size,
                max_rollbacks=args.max_rollbacks,
                total_time_limit_s=total_time_limit_s,
                allow_a2_fallback=not args.disable_a2_fallback,
                fast_large_instance_mode=is_large_instance,
                diagnostics_enabled=True,
            )
            output_path = args.output_dir / f"{instance_path.stem}.solution.json"
            output_path.write_text(json.dumps(result["solution"], indent=2), encoding="utf-8")
            runtime = round(time.time() - started_at, 2)
            budget_label = "none" if total_time_limit_s is None else f"{total_time_limit_s}s"
            print(
                f"[OK] {instance_path.name} | runtime={runtime}s budget={budget_label} objective={result['objective']} "
                f"tardiness={result['tardiness']} makespan={result['makespan']}"
            )
            successes += 1
        except Exception as exc:
            runtime = round(time.time() - started_at, 2)
            print(f"[FAIL] {instance_path.name} | runtime={runtime}s {type(exc).__name__}: {exc}")
            failures += 1

    print()
    print(f"Summary: successes={successes}, failures={failures}, total={len(instance_paths)}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
