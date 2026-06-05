import json
from typing import List, Dict, Tuple, Set

class PMSInstance:
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.n_jobs = len(data["Jobs"])
        self.n_machines = len(data["Machines"])
        self.n_resources = len(data.get("Resources", []))
        
        self.processing = [j["ProcessingTime"] for j in data["Jobs"]]
        self.due = [j["DueTime"] for j in data["Jobs"]]
        self.initial_setup = [j["InitialSetupTime"] for j in data["Jobs"]]
        
        #List of sets of eligible machines for each job
        self.eligible: List[Set[int]] = [set(j["EligibleMachineIds"]) for j in data["Jobs"]]
        #List of sets of precedence job ids for each job
        self.precedences: List[Set[int]] = [set(j["PrecedenceJobIds"]) for j in data["Jobs"]]
        #List of sets of setup times for each job
        self.setup: List[List[int]] = [j["JobSetupTimes"] for j in data["Jobs"]]
        # setup[j][k] = setup time when job j follows job k (0-based)
        #self.setup = [j["JobSetupTimes"] for j in data["Jobs"]]

        self.job_resource_requirements : List[List[Dict[str,int]]] = [j["RequiredResources"] for j in data["Jobs"]]
        self.num_resource_periods: List[int] = [len(resource.get("AvailabilityPeriods", []))
                for resource in data.get("Resources", [])]
        self.resource_availability_starts : List[List[int]]  = [
            [period["Start"] for period in resource.get("AvailabilityPeriods", [])]
            for resource in data.get("Resources", [])]

        self.resource_availability_ends : List[List[int]]  = [
            [period["End"] for period in resource.get("AvailabilityPeriods", [])]
            for resource in data.get("Resources", [])]

        self.resource_availability_capacities : List[List[int]]  = [
            [period["Capacity"] for period in resource.get("AvailabilityPeriods", [])]
            for resource in data.get("Resources", [])]
        #RESOURCES
        # self.resources = data.get("Resources", [])
        # self.required_resources = []
        # for j in data["Jobs"]:
        #     req = [0] * max(1, self.n_resources)
        #     for r in j.get("RequiredResources", []):
        #         if r["ResourceId"] <= len(req):
        #             req[r["ResourceId"]-1] = r["Capacity"]
        #     self.required_resources.append(req)

        '''"Resources": [
        {
            "Id": 1,
            "Name": "Resource 1",
            "AvailabilityPeriods": [
                {
                    "Start": 0,
                    "End": 1709,
                    "Capacity": 11
                },
        "RequiredResources": [
                {
                    "ResourceId": 1,
                    "Capacity": 4
                }
            ],        
                
        '''

    
        '''
        Keep job_sequences_by_machine.
        Keep next_job_pos[machine_id] for the next unscheduled job on each machine.
        Repeatedly scan all machines.
        For each machine, look only at its next job.
        Schedule that job only if all predecessors already have finish_times.
        Repeat until all jobs are scheduled, or no progress is possible.'''
        #--2.NON-OVERLAPPING JOBS--
        
        
        #--6.EXTRAS--
        #TODO: no detection of precedence cycles
        #TODO no explicit validation of machine id range

    #machine assignment - list of machine ids for each job (1-based)
    def decode(self, machine_assignment: List[int]) -> Tuple[int, int, int, bool, Dict]:
        """Decode machine assignment into full schedule.
        Returns (objective, tardiness, makespan, feasible, solution_dict)"""
        #--1.MACHINE ELIGIBILITY--
        job_sequences_by_machine = [[] for _ in range(self.n_machines + 1)]
        for job_id in range(self.n_jobs):
            machine_id = machine_assignment[job_id]
            if machine_id not in self.eligible[job_id]:
                return 10**9, 10**9, 10**9, False, {}
            job_sequences_by_machine[machine_id].append(job_id)
        
        start_times = [0] * self.n_jobs
        finish_times = [0] * self.n_jobs
        makespan = 0
        tardiness = 0


        scheduled_jobs = set()
        next_job_pos = [0] * (self.n_machines + 1)
        machine_ready_time = [0] * (self.n_machines + 1)
        previous_job_by_machine = [None] * (self.n_machines + 1)
        
        while len(scheduled_jobs) < self.n_jobs:
            progress = False
            
            for machine_id in range(1, self.n_machines + 1):
                job_sequence = job_sequences_by_machine[machine_id]
                job_pos = next_job_pos[machine_id]

                if job_pos >= len(job_sequence):
                    continue

                job_id = job_sequence[job_pos]

                    
                machine_time = machine_ready_time[machine_id]
                previous_job_id = previous_job_by_machine[machine_id]
                has_unscheduled_predecessor = False

                #--3.JOB PRECEDENCES--
                predecessor_indices = [pred_job_id - 1 for pred_job_id in self.precedences[job_id]]

                for predecessor_job_id in predecessor_indices:
                    if predecessor_job_id not in scheduled_jobs:
                        has_unscheduled_predecessor = True
                        break

                if has_unscheduled_predecessor: continue
                
                #--4.SETUP TIMES--
                if previous_job_id is None:
                    machine_time = max(machine_time, self.initial_setup[job_id])
                else:
                    machine_time += self.setup[job_id][previous_job_id]
                
                latest_predecessor_finish = max(
                        (finish_times[predecessor_job_id] for predecessor_job_id in predecessor_indices),
                        default=0,
                )
                machine_time = max(machine_time, latest_predecessor_finish)

                #--5.RESOURCE MANAGEMENT--
                '''
                array[RESOURCES] of set of JOBS: jobs_using_resource =
                [ { j | j in JOBS where required_resource[j][r] > 0 } | r in RESOURCES ];

                constraint
                forall(r in RESOURCES, p in 1..n_res_periods[r]) (
                    sum(j in jobs_using_resource[r]) (
                        required_resource[j][r] *
                        bool2int(start[j] < res_period_end[r][p] /\ end[j] > res_period_start[r][p])
                    )
                    <= res_period_capacity[r][p]
                );
                '''
                has_resource_conflict = False

                '''
                "RequiredResources": [
                    {
                        "ResourceId": 1,
                        "Capacity": 4
                    }
                ],'''

                candidate_start = machine_time
                candidate_finish = candidate_start + self.processing[job_id]

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

                            overlaps_interval = (scheduled_start < overlap_end and overlap_start < scheduled_end)
                            if not overlaps_interval:
                                continue
                            
                            for req in self.job_resource_requirements[scheduled_job_id]:
                                if req["ResourceId"] - 1 == resource_id:
                                    used_capacity += req["Capacity"]
                                    break

                        if used_capacity > period_capacity:
                            has_resource_conflict = True
                            break
                    
                    if has_resource_conflict:
                        break

                if has_resource_conflict:
                    continue


                




                start_times[job_id] = machine_time
                finish_times[job_id] = start_times[job_id] + self.processing[job_id]
                machine_time = finish_times[job_id]
                previous_job_id = job_id

                makespan = max(makespan, machine_time)

                next_job_pos[machine_id] += 1     
                machine_ready_time[machine_id] = machine_time
                previous_job_by_machine[machine_id] = previous_job_id
                scheduled_jobs.add(job_id)
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
                    "MachineId": int(machine_assignment[job_id])
                }
                for job_id in range(self.n_jobs)
            ]
        }
        
        return objective, tardiness, makespan, True, solution

    def is_feasible(self, machine_assignment: List[int]) -> bool:
        _, _, _, feasible, _ = self.decode(machine_assignment)
        return feasible
