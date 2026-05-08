from __future__ import annotations

import argparse
import json
import queue
import re
import subprocess
import threading
import time
from pathlib import Path


INSTANCE_RE = re.compile(r"^PSSAI_PMS_(.+)\.dzn$")
JOB_COUNT_RE = re.compile(r"_j(\d+)_")


class NoSolutionFound(Exception):
    pass


def extract_last_json_block(output: str) -> str:
    for block in reversed(re.split(r"(?m)^-{10,}$|^={10,}$|^=====UNKNOWN=====$", output)):
        start = block.find("{")
        end = block.rfind("}")
        if start == -1 or end == -1 or end < start:
            continue

        text = block[start : end + 1]
        try:
            json.loads(text)
        except json.JSONDecodeError:
            continue
        return text

    raise NoSolutionFound(
        "MiniZinc did not print any valid JSON solution. "
        "The instance may need a longer time limit before the first feasible solution is found."
    )


def solution_name(instance: Path) -> str:
    return f"{instance.stem}.solution.json"


def job_count(instance: Path) -> int:
    match = JOB_COUNT_RE.search(instance.name)
    if not match:
        raise ValueError(f"Cannot determine job count from filename: {instance.name}")
    return int(match.group(1))


def time_limit_for_instance(instance: Path) -> int:
    jobs = job_count(instance)
    if 10 <= jobs <= 50:
        return 20 * 60 * 1000
    if jobs == 100:
        return  5 * 30 * 60 * 1000
    if jobs == 500:
        return 4 * 60 * 60 * 1000
    raise ValueError(f"No time-limit rule configured for {jobs} jobs in {instance.name}")


def format_duration(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def save_last_solution(output: str, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(extract_last_json_block(output) + "\n", encoding="utf-8")


def read_stream(stream, lines: queue.Queue[str]) -> None:
    try:
        for line in stream:
            lines.put(line)
    finally:
        stream.close()


def run_instance(
    minizinc: Path,
    model: Path,
    instance: Path,
    output_file: Path,
    timeout_ms: int,
    gap: float | None,
    threads: int,
    solver: str,
) -> str:
    cmd = [
        str(minizinc),
        "--solver",
        solver,
        "--time-limit",
        str(timeout_ms),
        "--parallel",
        str(threads),
        "--intermediate",
        str(model),
        str(instance),
    ]
    if gap is not None:
        cmd[cmd.index(str(model)):cmd.index(str(model))] = [
            "--fzn-flag",
            f"--params=relative_gap_limit:{gap}",
        ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(target=read_stream, args=(process.stdout, lines), daemon=True)
    reader.start()

    output_parts: list[str] = []
    saved_any = False
    deadline = time.monotonic() + (timeout_ms / 1000)

    def try_save() -> bool:
        try:
            save_last_solution("".join(output_parts), output_file)
            return True
        except NoSolutionFound:
            return False

    def stop_process() -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def drain_lines() -> None:
        while True:
            try:
                output_parts.append(lines.get_nowait())
            except queue.Empty:
                break

    try:
        while True:
            if process.poll() is not None:
                drain_lines()
                break

            if time.monotonic() >= deadline:
                stop_process()
                drain_lines()

                if try_save():
                    return "timeout_saved"
                raise NoSolutionFound(
                    "MiniZinc timed out before printing any valid JSON solution."
                )

            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue

            output_parts.append(line)
            if line.strip() in {"----------", "==========", "=====UNKNOWN====="}:
                saved_any = try_save() or saved_any
    except KeyboardInterrupt:
        stop_process()
        drain_lines()
        if try_save():
            return "interrupted_saved"
        return "interrupted_no_solution"

    reader.join(timeout=1)
    combined_output = "".join(output_parts)
    returncode = process.returncode
    if returncode not in (0, 1):
        raise RuntimeError(combined_output.strip())

    if try_save():
        if "=====UNKNOWN=====" in combined_output:
            return "timeout_saved"
        return "saved"

    if saved_any:
        return "saved"
    raise NoSolutionFound(
        "MiniZinc did not print any valid JSON solution."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minizinc", default=r"C:\Program Files\MiniZinc\minizinc.exe")
    parser.add_argument("--model", default="job_scheduling.mzn")
    parser.add_argument("--instances", default="conv_instances")
    parser.add_argument("--solutions", default="results")
    parser.add_argument("--time-limit", type=int, default=900000)
    parser.add_argument("--fixed-time-limit", action="store_true")
    parser.add_argument("--gap", type=float, default=0.05)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--solver", default="OR Tools CP-SAT")
    parser.add_argument("--pattern", default="*.dzn")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    minizinc = Path(args.minizinc)
    model = (root / args.model).resolve()
    instances_dir = (root / args.instances).resolve()
    solutions_dir = (root / args.solutions).resolve()

    instances = sorted(instances_dir.rglob(args.pattern), key=lambda path: (job_count(path), path.name))
    if args.limit > 0:
        instances = instances[: args.limit]

    if not instances:
        raise SystemExit(f"No .dzn files found in {instances_dir}")

    failures = []
    no_solution = []
    timed_out_saved = []
    saved = 0
    for index, instance in enumerate(instances, start=1):
        relative_parent = instance.parent.relative_to(instances_dir)
        output_file = solutions_dir / relative_parent / solution_name(instance)
        instance_time_limit = args.time_limit if args.fixed_time_limit else time_limit_for_instance(instance)
        jobs = job_count(instance)
        instance_gap = None if jobs == 10 else args.gap
        gap_text = "none" if instance_gap is None else str(instance_gap)
        print(
            f"[{index}/{len(instances)}] {instance.relative_to(instances_dir)} -> "
            f"{output_file.relative_to(solutions_dir)} "
            f"(time limit: {format_duration(instance_time_limit)}, gap: {gap_text})",
            flush=True,
        )

        try:
            status = run_instance(
                minizinc=minizinc,
                model=model,
                instance=instance,
                output_file=output_file,
                timeout_ms=instance_time_limit,
                gap=instance_gap,
                threads=args.threads,
                solver=args.solver,
            )
            saved += 1
            if status == "timeout_saved":
                timed_out_saved.append(instance)
                print(f"  TIMED OUT, SAVED BEST FOUND: {output_file}", flush=True)
            elif status == "interrupted_saved":
                timed_out_saved.append(instance)
                print(f"  INTERRUPTED, SAVED BEST FOUND: {output_file}", flush=True)
            elif status == "interrupted_no_solution":
                saved -= 1
                print("  INTERRUPTED, NO SOLUTION SAVED", flush=True)
            else:
                print(f"  SAVED: {output_file}", flush=True)
        except KeyboardInterrupt:
            print("\nStopped by user. Already written solution files were left in place.")
            raise SystemExit(130)
        except NoSolutionFound as exc:
            no_solution.append((instance, exc))
            print(f"  NO SOLUTION SAVED: {exc}", flush=True)
        except Exception as exc:
            failures.append((instance, exc))
            print(f"  FAILED: {exc}", flush=True)

    print(f"\nDone. Saved {saved}/{len(instances)} solution files to {solutions_dir}")

    if no_solution:
        print("\nNo solution found before timeout:")
        for instance, exc in no_solution:
            print(f"- {instance}: {exc}")

    if timed_out_saved:
        print("\nTimed out but saved best incumbent:")
        for instance in timed_out_saved:
            print(f"- {instance}")

    if failures:
        print("\nSolver/model errors:")
        for instance, exc in failures:
            print(f"- {instance}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
