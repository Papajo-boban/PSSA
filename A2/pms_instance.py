import json
from typing import Dict, List, Set, Tuple


class PMSInstance:
    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)

        self.n_jobs = len(data["Jobs"])
        self.n_machines = len(data["Machines"])
        self.n_resources = len(data.get("Resources", []))

        self.processing = [job["ProcessingTime"] for job in data["Jobs"]]
        self.due = [job["DueTime"] for job in data["Jobs"]]
        self.initial_setup = [job["InitialSetupTime"] for job in data["Jobs"]]

        # Input IDs are 1-based.
        self.eligible: List[List[int]] = [(job["EligibleMachineIds"]) for job in data["Jobs"]]
        self.predecessor_indices: List[List[int]] = [
            [pred_job_id - 1 for pred_job_id in job["PrecedenceJobIds"]]
            for job in data["Jobs"]
        ]

        # setup[j][k] = setup time when job j follows job k (0-based job index)
        self.setup: List[List[int]] = [job["JobSetupTimes"] for job in data["Jobs"]]

        self.job_resource_requirements: List[List[Dict[str, int]]] = [
            job["RequiredResources"] for job in data["Jobs"]
        ]
        self.num_resource_periods: List[int] = [
            len(resource.get("AvailabilityPeriods", []))
            for resource in data.get("Resources", [])
        ]
        self.resource_availability_starts: List[List[int]] = [
            [period["Start"] for period in resource.get("AvailabilityPeriods", [])]
            for resource in data.get("Resources", [])
        ]
        self.resource_availability_ends: List[List[int]] = [
            [period["End"] for period in resource.get("AvailabilityPeriods", [])]
            for resource in data.get("Resources", [])
        ]
        self.resource_availability_capacities: List[List[int]] = [
            [period["Capacity"] for period in resource.get("AvailabilityPeriods", [])]
            for resource in data.get("Resources", [])
        ]

    def decode(self, job_sequences_by_machine: List[List[int]]) -> Tuple[int, int, int, bool, Dict]:
        """Decode per-machine job sequences into a concrete schedule.

        Returns (objective, tardiness, makespan, feasible, solution_dict).
        """
        machine_assignment = [0] * self.n_jobs
        seen_jobs = set()

        if len(job_sequences_by_machine) != self.n_machines + 1:
            return 10**9, 10**9, 10**9, False, {}

        for machine_id in range(1, self.n_machines + 1):
            for job_id in job_sequences_by_machine[machine_id]:
                if job_id < 0 or job_id >= self.n_jobs:
                    return 10**9, 10**9, 10**9, False, {}
                if job_id in seen_jobs:
                    return 10**9, 10**9, 10**9, False, {}
                if machine_id not in self.eligible[job_id]:
                    return 10**9, 10**9, 10**9, False, {}

                seen_jobs.add(job_id)
                machine_assignment[job_id] = machine_id

        if len(seen_jobs) != self.n_jobs:
            return 10**9, 10**9, 10**9, False, {}

        start_times = [0] * self.n_jobs
        finish_times = [0] * self.n_jobs
        scheduled_jobs = set()
        next_job_pos = [0] * (self.n_machines + 1)
        machine_ready_time = [0] * (self.n_machines + 1)
        previous_job_by_machine = [None] * (self.n_machines + 1)
        makespan = 0

        while len(scheduled_jobs) < self.n_jobs:
            progress = False

            for machine_id in range(1, self.n_machines + 1):
                job_sequence = job_sequences_by_machine[machine_id]
                job_pos = next_job_pos[machine_id]

                if job_pos >= len(job_sequence):
                    continue

                job_id = job_sequence[job_pos]
                predecessor_indices = self.predecessor_indices[job_id]

                if any(predecessor_job_id not in scheduled_jobs for predecessor_job_id in predecessor_indices):
                    continue

                machine_time = machine_ready_time[machine_id]
                previous_job_id = previous_job_by_machine[machine_id]

                # Apply setup first, then wait for latest predecessor if needed.
                if previous_job_id is None:
                    machine_time = max(machine_time, self.initial_setup[job_id])
                else:
                    machine_time += self.setup[job_id][previous_job_id]

                latest_predecessor_finish = max(
                    (finish_times[predecessor_job_id] for predecessor_job_id in predecessor_indices),
                    default=0,
                )
                machine_time = max(machine_time, latest_predecessor_finish)

                candidate_start = machine_time
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
                return 10**9, 10**9, 10**9, False, {}

        tardiness = sum(max(0, finish_times[job_id] - self.due[job_id]) for job_id in range(self.n_jobs))
        objective = tardiness + makespan

        solution = {
            "Jobs": [
                {
                    "JobId": job_id + 1,
                    "StartTime": int(start_times[job_id]),
                    "MachineId": int(machine_assignment[job_id]),
                }
                for job_id in range(self.n_jobs)
            ]
        }

        return objective, tardiness, makespan, True, solution

    def has_resource_conflict(
        self,
        job_id: int,
        candidate_start: int,
        candidate_finish: int,
        scheduled_jobs: Set[int],
        start_times: List[int],
        finish_times: List[int],
    ) -> bool:
        for requirement in self.job_resource_requirements[job_id]:
            resource_id = requirement["ResourceId"] - 1
            required_amount = requirement["Capacity"]

            for period_id in range(self.num_resource_periods[resource_id]):
                period_start = self.resource_availability_starts[resource_id][period_id]
                period_end = self.resource_availability_ends[resource_id][period_id]
                period_capacity = self.resource_availability_capacities[resource_id][period_id]

                overlap_start = max(candidate_start, period_start)
                overlap_end = min(candidate_finish, period_end)
                if overlap_start >= overlap_end:
                    continue

                used_capacity = required_amount
                for scheduled_job_id in scheduled_jobs:
                    scheduled_start = start_times[scheduled_job_id]
                    scheduled_end = finish_times[scheduled_job_id]
                    overlaps_interval = (
                        scheduled_start < overlap_end and overlap_start < scheduled_end
                    )
                    if not overlaps_interval:
                        continue

                    for scheduled_requirement in self.job_resource_requirements[scheduled_job_id]:
                        if scheduled_requirement["ResourceId"] - 1 == resource_id:
                            used_capacity += scheduled_requirement["Capacity"]
                            break

                if used_capacity > period_capacity:
                    return True

        return False
