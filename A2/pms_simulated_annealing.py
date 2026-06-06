import copy
import math
import random
import statistics
import time
from datetime import datetime
from typing import Dict, List

from pms_instance import PMSInstance


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


class SimulatedAnnealing:
    def __init__(self, instance: PMSInstance, **kwargs):
        self.instance = instance
        self.initial_temp = kwargs.get("initial_temp", 5000.0)
        self.cooling_rate = kwargs.get("cooling_rate", 0.995)
        self.min_temp = kwargs.get("min_temp", 0.01)
        self.iter_per_temp = kwargs.get("iter_per_temp", 300)
        self.max_runtime = kwargs.get("max_runtime", 60)

    def random_solution(self) -> List[List[int]]:
        """Generate random per-machine job sequences."""
        job_sequences_by_machine = [[] for _ in range(self.instance.n_machines + 1)]
        jobs = list(range(self.instance.n_jobs))
        random.shuffle(jobs)

        for job_id in jobs:
            eligible_machines = list(self.instance.eligible[job_id])
            machine_id = random.choice(eligible_machines)
            job_sequences_by_machine[machine_id].append(job_id)

        return job_sequences_by_machine

    def construct_feasible_solution(self, max_restarts: int = 300, candidate_pool_size: int = 3) -> List[List[int]]:
        """Construct a feasible schedule incrementally."""
        for _ in range(max_restarts):
            job_sequences_by_machine = [[] for _ in range(self.instance.n_machines + 1)]
            unscheduled_jobs = set(range(self.instance.n_jobs))
            scheduled_jobs = set()

            start_times = [0] * self.instance.n_jobs
            finish_times = [0] * self.instance.n_jobs
            machine_ready_time = [0] * (self.instance.n_machines + 1)
            previous_job_by_machine = [None] * (self.instance.n_machines + 1)

            while unscheduled_jobs:
                candidate_moves = []
                candidate_jobs = list(unscheduled_jobs)
                random.shuffle(candidate_jobs)

                for job_id in candidate_jobs:
                    predecessor_indices = self.instance.predecessor_indices[job_id]
                    if any(predecessor_job_id not in scheduled_jobs for predecessor_job_id in predecessor_indices):
                        continue

                    eligible_machines = list(self.instance.eligible[job_id])
                    random.shuffle(eligible_machines)

                    for machine_id in eligible_machines:
                        candidate_start = machine_ready_time[machine_id]
                        previous_job_id = previous_job_by_machine[machine_id]

                        if previous_job_id is None:
                            candidate_start = max(candidate_start, self.instance.initial_setup[job_id])
                        else:
                            candidate_start += self.instance.setup[job_id][previous_job_id]

                        latest_predecessor_finish = max(
                            (finish_times[pred_job_id] for pred_job_id in predecessor_indices),
                            default=0,
                        )
                        candidate_start = max(candidate_start, latest_predecessor_finish)
                        candidate_finish = candidate_start + self.instance.processing[job_id]

                        if self.instance.has_resource_conflict(
                            job_id,
                            candidate_start,
                            candidate_finish,
                            scheduled_jobs,
                            start_times,
                            finish_times,
                        ):
                            continue

                        candidate_moves.append((candidate_finish, candidate_start, job_id, machine_id))

                if not candidate_moves:
                    break

                candidate_moves.sort(key=lambda move: (move[0], move[1], move[2], move[3]))
                pool_size = min(candidate_pool_size, len(candidate_moves))
                chosen_finish, chosen_start, chosen_job_id, chosen_machine_id = random.choice(
                    candidate_moves[:pool_size]
                )

                job_sequences_by_machine[chosen_machine_id].append(chosen_job_id)
                start_times[chosen_job_id] = chosen_start
                finish_times[chosen_job_id] = chosen_finish
                machine_ready_time[chosen_machine_id] = chosen_finish
                previous_job_by_machine[chosen_machine_id] = chosen_job_id
                scheduled_jobs.add(chosen_job_id)
                unscheduled_jobs.remove(chosen_job_id)

            if not unscheduled_jobs:
                return job_sequences_by_machine

        raise RuntimeError("Could not construct a feasible initial solution.")

    def get_neighbor(self, current: List[List[int]]) -> List[List[int]]:
        """Strong neighborhood: reorder or reassign jobs across machine sequences."""
        neighbor = copy.deepcopy(current)
        move = random.random()

        if move < 0.35:
            non_empty_machines = [m for m in range(1, self.instance.n_machines + 1) if neighbor[m]]
            if not non_empty_machines:
                return neighbor

            source_machine = random.choice(non_empty_machines)
            source_pos = random.randrange(len(neighbor[source_machine]))
            job_id = neighbor[source_machine].pop(source_pos)

            eligible_machines = [m for m in self.instance.eligible[job_id] if m != source_machine]
            if not eligible_machines:
                neighbor[source_machine].insert(source_pos, job_id)
                return neighbor

            target_machine = random.choice(eligible_machines)
            target_pos = random.randrange(len(neighbor[target_machine]) + 1)
            neighbor[target_machine].insert(target_pos, job_id)

        elif move < 0.70:
            candidate_machines = [m for m in range(1, self.instance.n_machines + 1) if len(neighbor[m]) >= 2]
            if not candidate_machines:
                return neighbor

            machine_id = random.choice(candidate_machines)
            i, j = random.sample(range(len(neighbor[machine_id])), 2)
            neighbor[machine_id][i], neighbor[machine_id][j] = neighbor[machine_id][j], neighbor[machine_id][i]

        else:
            candidate_machines = [m for m in range(1, self.instance.n_machines + 1) if len(neighbor[m]) >= 2]
            if not candidate_machines:
                return neighbor

            machine_id = random.choice(candidate_machines)
            from_pos, to_pos = random.sample(range(len(neighbor[machine_id])), 2)
            job_id = neighbor[machine_id].pop(from_pos)
            neighbor[machine_id].insert(to_pos, job_id)

        return neighbor

    def run(self) -> Dict:
        current = self.construct_feasible_solution()
        current_obj, current_tard, current_makespan, _, _ = self.instance.decode(current)

        best = copy.deepcopy(current)
        best_obj = current_obj
        best_tard = current_tard
        best_makespan = current_makespan

        temp = self.initial_temp
        start_time = time.time()
        iterations = 0
        print(f"[{timestamp()}] starting annealing phase ({self.max_runtime}s budget)...")

        while temp > self.min_temp and (time.time() - start_time) < self.max_runtime:
            for _ in range(self.iter_per_temp):
                iterations += 1
                neighbor = self.get_neighbor(current)
                n_obj, n_tard, n_makespan, feasible, _ = self.instance.decode(neighbor)

                if not feasible:
                    continue

                delta = n_obj - current_obj

                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current = neighbor
                    current_obj = n_obj
                    current_tard = n_tard
                    current_makespan = n_makespan

                    if current_obj < best_obj:
                        best = copy.deepcopy(current)
                        best_obj = current_obj
                        best_tard = n_tard
                        best_makespan = n_makespan

            temp *= self.cooling_rate

        _, _, _, _, solution_dict = self.instance.decode(best)

        return {
            "objective": int(best_obj),
            "tardiness": int(best_tard),
            "makespan": int(best_makespan),
            "solution": solution_dict,
            "iterations": iterations,
            "runtime": round(time.time() - start_time, 2),
        }


def run_multiple_times(instance_path: str, n_runs: int = 5, **sa_params):
    instance = PMSInstance(instance_path)
    results = []

    print(f"Running SA on {instance_path.split('/')[-1]} ({n_runs} runs)...")

    for run in range(1, n_runs + 1):
        print(f"  Run {run}/{n_runs}...", end=" ")
        sa = SimulatedAnnealing(instance, **sa_params)
        result = sa.run()
        results.append(result)
        print(f"Obj = {result['objective']}")

    objs = [r["objective"] for r in results]

    summary = {
        "instance": instance_path.split("/")[-1],
        "best_objective": min(objs),
        "avg_objective": round(statistics.mean(objs), 1),
        "std_objective": round(statistics.stdev(objs), 2) if n_runs > 1 else 0,
        "results": results,
    }

    print(f"Best: {summary['best_objective']} | Avg: {summary['avg_objective']} +/- {summary['std_objective']}")
    return summary