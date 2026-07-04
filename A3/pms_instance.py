import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


class PMSInstance:
    def __init__(self, json_path: str):
        self.json_path = json_path
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))

        self.n_jobs = len(data["Jobs"])
        self.n_machines = len(data["Machines"])
        self.n_resources = len(data.get("Resources", []))

        self.processing = [job["ProcessingTime"] for job in data["Jobs"]]
        self.due = [job["DueTime"] for job in data["Jobs"]]
        self.initial_setup = [job["InitialSetupTime"] for job in data["Jobs"]]
        self.eligible: List[List[int]] = [list(job["EligibleMachineIds"]) for job in data["Jobs"]]
        self.predecessor_indices: List[List[int]] = [
            [pred_job_id - 1 for pred_job_id in job["PrecedenceJobIds"]]
            for job in data["Jobs"]
        ]
        self.setup: List[List[int]] = [job["JobSetupTimes"] for job in data["Jobs"]]

        self.job_resource_requirements: List[List[Tuple[int, int]]] = [
            [(req["ResourceId"] - 1, req["Capacity"]) for req in job["RequiredResources"]]
            for job in data["Jobs"]
        ]
        self.resource_periods: List[List[Tuple[int, int, int]]] = [
            [
                (period["Start"], period["End"], period["Capacity"])
                for period in resource.get("AvailabilityPeriods", [])
            ]
            for resource in data.get("Resources", [])
        ]

        self.successor_indices: List[List[int]] = [[] for _ in range(self.n_jobs)]
        for job_id, predecessors in enumerate(self.predecessor_indices):
            for predecessor_id in predecessors:
                self.successor_indices[predecessor_id].append(job_id)

        self.machine_job_counts = [0] * (self.n_machines + 1)
        for eligible_machines in self.eligible:
            for machine_id in eligible_machines:
                self.machine_job_counts[machine_id] += 1

        self.resource_weights = [
            sum(capacity for _, capacity in requirements)
            for requirements in self.job_resource_requirements
        ]

    def job_priority_key(self, job_id: int) -> Tuple[int, int, int, int, int]:
        slack = self.due[job_id] - self.processing[job_id]
        return (
            len(self.eligible[job_id]),
            slack,
            -self.resource_weights[job_id],
            -len(self.successor_indices[job_id]),
            -self.processing[job_id],
        )

    def solution_from_schedule(self, start_times: List[int], machine_assignment: List[int]) -> Dict:
        return {
            "Jobs": [
                {
                    "JobId": job_id + 1,
                    "StartTime": int(start_times[job_id]),
                    "MachineId": int(machine_assignment[job_id]),
                }
                for job_id in range(self.n_jobs)
            ]
        }

    def has_resource_conflict(
        self,
        job_id: int,
        candidate_start: int,
        candidate_finish: int,
        scheduled_jobs: Set[int],
        start_times: List[int],
        finish_times: List[int],
    ) -> bool:
        for resource_id, required_amount in self.job_resource_requirements[job_id]:
            for period_start, period_end, period_capacity in self.resource_periods[resource_id]:
                overlap_start = max(candidate_start, period_start)
                overlap_end = min(candidate_finish, period_end)
                if overlap_start >= overlap_end:
                    continue

                used_capacity = required_amount
                for scheduled_job_id in scheduled_jobs:
                    scheduled_start = start_times[scheduled_job_id]
                    scheduled_end = finish_times[scheduled_job_id]
                    if not (scheduled_start < overlap_end and overlap_start < scheduled_end):
                        continue

                    for scheduled_resource_id, scheduled_amount in self.job_resource_requirements[scheduled_job_id]:
                        if scheduled_resource_id == resource_id:
                            used_capacity += scheduled_amount
                            break

                if used_capacity > period_capacity:
                    return True
        return False

    def earliest_resource_feasible_start(
        self,
        job_id: int,
        earliest_start: int,
        scheduled_jobs: Set[int],
        start_times: List[int],
        finish_times: List[int],
    ) -> int:
        start = earliest_start
        duration = self.processing[job_id]

        while True:
            finish = start + duration
            shifted = False

            for resource_id, required_amount in self.job_resource_requirements[job_id]:
                periods = self.resource_periods[resource_id]
                covering_period = None
                for period_start, period_end, period_capacity in periods:
                    if period_start <= start and finish <= period_end:
                        covering_period = (period_start, period_end, period_capacity)
                        break
                    if start < period_start:
                        start = period_start
                        shifted = True
                        break
                if shifted:
                    break
                if covering_period is None:
                    future_periods = [period for period in periods if period[0] >= start]
                    if not future_periods:
                        return 10**9
                    start = future_periods[0][0]
                    shifted = True
                    break

                _, period_end, period_capacity = covering_period
                if finish > period_end:
                    start = period_end
                    shifted = True
                    break

                conflict_end = None
                used_capacity = required_amount
                overlapping_jobs = []
                for scheduled_job_id in scheduled_jobs:
                    scheduled_start = start_times[scheduled_job_id]
                    scheduled_end = finish_times[scheduled_job_id]
                    if not (scheduled_start < finish and start < scheduled_end):
                        continue
                    for scheduled_resource_id, scheduled_amount in self.job_resource_requirements[scheduled_job_id]:
                        if scheduled_resource_id == resource_id:
                            used_capacity += scheduled_amount
                            overlapping_jobs.append((scheduled_end, scheduled_amount))
                            break

                if used_capacity > period_capacity:
                    overlapping_jobs.sort()
                    running_capacity = used_capacity
                    for scheduled_end, scheduled_amount in overlapping_jobs:
                        running_capacity -= scheduled_amount
                        if running_capacity <= period_capacity:
                            conflict_end = scheduled_end
                            break
                    start = max(start + 1, conflict_end if conflict_end is not None else finish)
                    shifted = True
                    break

            if not shifted:
                return start

    def decode_sequences(
        self, machine_sequences: List[List[int]]
    ) -> Tuple[int, int, int, bool, Dict, List[int], List[int], List[int]]:
        machine_assignment = [0] * self.n_jobs
        seen_jobs = set()

        if len(machine_sequences) != self.n_machines + 1:
            return 10**9, 10**9, 10**9, False, {}, [], [], []

        for machine_id in range(1, self.n_machines + 1):
            for job_id in machine_sequences[machine_id]:
                if job_id < 0 or job_id >= self.n_jobs:
                    return 10**9, 10**9, 10**9, False, {}, [], [], []
                if job_id in seen_jobs:
                    return 10**9, 10**9, 10**9, False, {}, [], [], []
                if machine_id not in self.eligible[job_id]:
                    return 10**9, 10**9, 10**9, False, {}, [], [], []
                seen_jobs.add(job_id)
                machine_assignment[job_id] = machine_id

        if len(seen_jobs) != self.n_jobs:
            return 10**9, 10**9, 10**9, False, {}, [], [], []

        start_times = [0] * self.n_jobs
        finish_times = [0] * self.n_jobs
        scheduled_jobs: Set[int] = set()
        next_job_pos = [0] * (self.n_machines + 1)
        machine_ready_time = [0] * (self.n_machines + 1)
        previous_job_by_machine = [None] * (self.n_machines + 1)
        makespan = 0

        while len(scheduled_jobs) < self.n_jobs:
            progress = False
            for machine_id in range(1, self.n_machines + 1):
                job_pos = next_job_pos[machine_id]
                sequence = machine_sequences[machine_id]
                if job_pos >= len(sequence):
                    continue

                job_id = sequence[job_pos]
                predecessors = self.predecessor_indices[job_id]
                if any(pred_id not in scheduled_jobs for pred_id in predecessors):
                    continue

                machine_time = machine_ready_time[machine_id]
                previous_job_id = previous_job_by_machine[machine_id]
                if previous_job_id is None:
                    machine_time = max(machine_time, self.initial_setup[job_id])
                else:
                    machine_time += self.setup[job_id][previous_job_id]

                predecessor_finish = max((finish_times[pred_id] for pred_id in predecessors), default=0)
                candidate_start = max(machine_time, predecessor_finish)
                candidate_finish = candidate_start + self.processing[job_id]

                if self.has_resource_conflict(
                    job_id,
                    candidate_start,
                    candidate_finish,
                    scheduled_jobs,
                    start_times,
                    finish_times,
                ):
                    continue

                start_times[job_id] = candidate_start
                finish_times[job_id] = candidate_finish
                machine_ready_time[machine_id] = candidate_finish
                previous_job_by_machine[machine_id] = job_id
                next_job_pos[machine_id] += 1
                scheduled_jobs.add(job_id)
                makespan = max(makespan, candidate_finish)
                progress = True

            if not progress:
                return 10**9, 10**9, 10**9, False, {}, [], [], []

        tardiness = sum(max(0, finish_times[job_id] - self.due[job_id]) for job_id in range(self.n_jobs))
        solution = self.solution_from_schedule(start_times, machine_assignment)
        return tardiness + makespan, tardiness, makespan, True, solution, start_times, finish_times, machine_assignment

    def schedule_partial_sequences(
        self, machine_sequences: List[List[int]]
    ) -> Tuple[bool, List[int], List[int], List[int]]:
        machine_assignment = [0] * self.n_jobs
        seen_jobs = set()

        if len(machine_sequences) != self.n_machines + 1:
            return False, [], [], []

        for machine_id in range(1, self.n_machines + 1):
            for job_id in machine_sequences[machine_id]:
                if job_id < 0 or job_id >= self.n_jobs:
                    return False, [], [], []
                if job_id in seen_jobs:
                    return False, [], [], []
                if machine_id not in self.eligible[job_id]:
                    return False, [], [], []
                seen_jobs.add(job_id)
                machine_assignment[job_id] = machine_id

        start_times = [-1] * self.n_jobs
        finish_times = [-1] * self.n_jobs
        scheduled_jobs: Set[int] = set()
        target_jobs = len(seen_jobs)
        next_job_pos = [0] * (self.n_machines + 1)
        machine_ready_time = [0] * (self.n_machines + 1)
        previous_job_by_machine = [None] * (self.n_machines + 1)

        while len(scheduled_jobs) < target_jobs:
            progress = False
            for machine_id in range(1, self.n_machines + 1):
                job_pos = next_job_pos[machine_id]
                sequence = machine_sequences[machine_id]
                if job_pos >= len(sequence):
                    continue

                job_id = sequence[job_pos]
                predecessors = self.predecessor_indices[job_id]
                if any(
                    pred_id not in scheduled_jobs and pred_id in seen_jobs
                    for pred_id in predecessors
                ):
                    continue

                machine_time = machine_ready_time[machine_id]
                previous_job_id = previous_job_by_machine[machine_id]
                if previous_job_id is None:
                    machine_time = max(machine_time, self.initial_setup[job_id])
                else:
                    machine_time += self.setup[job_id][previous_job_id]

                predecessor_finish = max(
                    (finish_times[pred_id] for pred_id in predecessors if pred_id in seen_jobs),
                    default=0,
                )
                candidate_start = max(machine_time, predecessor_finish)
                candidate_finish = candidate_start + self.processing[job_id]

                if self.has_resource_conflict(
                    job_id,
                    candidate_start,
                    candidate_finish,
                    scheduled_jobs,
                    start_times,
                    finish_times,
                ):
                    continue

                start_times[job_id] = candidate_start
                finish_times[job_id] = candidate_finish
                machine_ready_time[machine_id] = candidate_finish
                previous_job_by_machine[machine_id] = job_id
                next_job_pos[machine_id] += 1
                scheduled_jobs.add(job_id)
                progress = True

            if not progress:
                return False, [], [], []

        return True, start_times, finish_times, machine_assignment
