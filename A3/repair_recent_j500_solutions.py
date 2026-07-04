import argparse
import json
import time
from pathlib import Path
from typing import List

from a3_initial_solution import InitialSolutionBuilder
from pms_instance import PMSInstance


def sequences_from_solution(instance: PMSInstance, solution_data: dict) -> List[List[int]]:
    jobs = solution_data.get("Jobs", [])
    machine_sequences = [[] for _ in range(instance.n_machines + 1)]
    seen = set()

    for machine_id in range(1, instance.n_machines + 1):
        scheduled = sorted(
            [job for job in jobs if job["MachineId"] == machine_id],
            key=lambda job: (job["StartTime"], job["JobId"]),
        )
        for job in scheduled:
            job_id = job["JobId"] - 1
            machine_sequences[machine_id].append(job_id)
            seen.add(job_id)

    missing_jobs = [job_id for job_id in range(instance.n_jobs) if job_id not in seen]
    for job_id in missing_jobs:
        machine_sequences[instance.eligible[job_id][0]].append(job_id)

    return machine_sequences


def main() -> int:
    parser = argparse.ArgumentParser(description="One-off repair for recently generated j500 solutions.")
    parser.add_argument("--instances-dir", type=Path, default=Path("instances"))
    parser.add_argument("--solutions-dir", type=Path, default=Path("solutions"))
    parser.add_argument("--minutes", type=int, default=240, help="How recent solution files must be.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of recent j500 files to repair.")
    parser.add_argument("--frontier-size", type=int, default=8)
    parser.add_argument("--rollback-size", type=int, default=6)
    parser.add_argument("--max-rollbacks", type=int, default=80)
    args = parser.parse_args()

    cutoff = time.time() - args.minutes * 60
    candidate_paths = sorted(
        (
            path for path in args.solutions_dir.glob("PSSAI_PMS_j500*.solution.json")
            if path.stat().st_mtime >= cutoff and ".repaired." not in path.name
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[: args.limit]

    if not candidate_paths:
        print("No recent j500 solution files found.")
        return 1

    for solution_path in candidate_paths:
        instance_stem = solution_path.name.split(".solution", 1)[0]
        instance_name = f"{instance_stem}.json"
        instance_path = args.instances_dir / instance_name
        if not instance_path.exists():
            print(f"[SKIP] {solution_path.name} | missing instance file {instance_name}")
            continue

        started_at = time.time()
        try:
            instance = PMSInstance(str(instance_path))
            solution_data = json.loads(solution_path.read_text(encoding="utf-8"))
            initial_sequences = sequences_from_solution(instance, solution_data)
            builder = InitialSolutionBuilder(
                instance,
                frontier_size=args.frontier_size,
                rollback_size=args.rollback_size,
                max_rollbacks=args.max_rollbacks,
                random_seed=0,
            )
            result = builder.large_instance_repair_lns(deadline=None, initial_sequences=initial_sequences)
            output_path = solution_path.with_name(solution_path.stem + ".repaired.solution.json")
            output_path.write_text(json.dumps(result["solution"], indent=2), encoding="utf-8")
            runtime = round(time.time() - started_at, 2)
            print(
                f"[OK] {solution_path.name} -> {output_path.name} | runtime={runtime}s "
                f"feasible={result.get('feasible')} infeasibility={result.get('infeasibility_score')} "
                f"objective={result['objective']}"
            )
        except Exception as exc:
            runtime = round(time.time() - started_at, 2)
            print(f"[FAIL] {solution_path.name} | runtime={runtime}s {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
