import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from resource_timeline import ResourceTimeline


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

        self.job_resource_requirements: List[List[Tuple[int, int]]] = []
        for job in data["Jobs"]:
            merged_requirements: Dict[int, int] = {}
            for requirement in job["RequiredResources"]:
                resource_id = requirement["ResourceId"] - 1
                merged_requirements[resource_id] = merged_requirements.get(resource_id, 0) + requirement["Capacity"]
            self.job_resource_requirements.append(sorted(merged_requirements.items()))
        self.resource_periods: List[List[Tuple[int, int, int]]] = [
            [
                (period["Start"], period["End"], period["Capacity"])
                for period in resource.get("AvailabilityPeriods", [])
            ]
            for resource in data.get("Resources", [])
        ]
        self.resource_timeline_periods = [
            list(periods)
            for periods in self.resource_periods
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
        resource_timelines = [
            ResourceTimeline(periods)
            for periods in self.resource_timeline_periods
        ]
        for scheduled_job_id in scheduled_jobs:
            scheduled_start = start_times[scheduled_job_id]
            scheduled_finish = finish_times[scheduled_job_id]
            for resource_id, required_amount in self.job_resource_requirements[scheduled_job_id]:
                resource_timelines[resource_id].commit(scheduled_start, scheduled_finish, required_amount)

        start = earliest_start
        duration = self.processing[job_id]
        while True:
            updated_start = start
            for resource_id, required_amount in self.job_resource_requirements[job_id]:
                feasible_start = resource_timelines[resource_id].earliest_feasible_start(
                    updated_start,
                    duration,
                    required_amount,
                )
                if feasible_start >= 10**9:
                    return 10**9
                updated_start = max(updated_start, feasible_start)
            if updated_start == start:
                return start
            start = updated_start

    def _build_resource_timelines(self) -> List[ResourceTimeline]:
        return [ResourceTimeline(periods) for periods in self.resource_timeline_periods]

    def _schedule_sequences_internal(
        self,
        machine_sequences: List[List[int]],
        allow_partial: bool,
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
        resource_timelines = self._build_resource_timelines()

        while len(scheduled_jobs) < target_jobs:
            progress = False
            for machine_id in range(1, self.n_machines + 1):
                job_pos = next_job_pos[machine_id]
                sequence = machine_sequences[machine_id]
                if job_pos >= len(sequence):
                    continue

                job_id = sequence[job_pos]
                predecessors = self.predecessor_indices[job_id]
                if allow_partial:
                    if any(pred_id not in scheduled_jobs and pred_id in seen_jobs for pred_id in predecessors):
                        continue
                else:
                    if any(pred_id not in scheduled_jobs for pred_id in predecessors):
                        continue

                machine_time = machine_ready_time[machine_id]
                previous_job_id = previous_job_by_machine[machine_id]
                if previous_job_id is None:
                    machine_time = max(machine_time, self.initial_setup[job_id])
                else:
                    machine_time += self.setup[job_id][previous_job_id]

                if allow_partial:
                    predecessor_finish = max(
                        (finish_times[pred_id] for pred_id in predecessors if pred_id in seen_jobs),
                        default=0,
                    )
                else:
                    predecessor_finish = max((finish_times[pred_id] for pred_id in predecessors), default=0)

                candidate_start = max(machine_time, predecessor_finish)
                while True:
                    updated_start = candidate_start
                    for resource_id, required_amount in self.job_resource_requirements[job_id]:
                        feasible_start = resource_timelines[resource_id].earliest_feasible_start(
                            updated_start,
                            self.processing[job_id],
                            required_amount,
                        )
                        if feasible_start >= 10**9:
                            return False, [], [], []
                        updated_start = max(updated_start, feasible_start)
                    if updated_start == candidate_start:
                        break
                    candidate_start = updated_start

                candidate_finish = candidate_start + self.processing[job_id]
                for resource_id, required_amount in self.job_resource_requirements[job_id]:
                    resource_timelines[resource_id].commit(candidate_start, candidate_finish, required_amount)

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

    def decode_sequences(
        self, machine_sequences: List[List[int]]
    ) -> Tuple[int, int, int, bool, Dict, List[int], List[int], List[int]]:
        feasible, start_times, finish_times, machine_assignment = self._schedule_sequences_internal(
            machine_sequences,
            allow_partial=False,
        )
        if not feasible:
            return 10**9, 10**9, 10**9, False, {}, [], [], []

        makespan = max(finish_times, default=0)
        tardiness = sum(max(0, finish_times[job_id] - self.due[job_id]) for job_id in range(self.n_jobs))
        solution = self.solution_from_schedule(start_times, machine_assignment)
        return tardiness + makespan, tardiness, makespan, True, solution, start_times, finish_times, machine_assignment

    def schedule_partial_sequences(
        self, machine_sequences: List[List[int]]
    ) -> Tuple[bool, List[int], List[int], List[int]]:
        return self._schedule_sequences_internal(machine_sequences, allow_partial=True)
