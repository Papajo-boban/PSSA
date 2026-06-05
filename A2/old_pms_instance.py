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
        self.setup: List[Set[int]] = [set(j["JobSetupTimes"]) for j in data["Jobs"]]
        # setup[j][k] = setup time when job j follows job k (0-based)
        #self.setup = [j["JobSetupTimes"] for j in data["Jobs"]]
        
        #RESOURCES
        # self.resources = data.get("Resources", [])
        # self.required_resources = []
        # for j in data["Jobs"]:
        #     req = [0] * max(1, self.n_resources)
        #     for r in j.get("RequiredResources", []):
        #         if r["ResourceId"] <= len(req):
        #             req[r["ResourceId"]-1] = r["Capacity"]
        #     self.required_resources.append(req)

    def decode(self, machine_assignment: List[int]) -> Tuple[int, int, int, bool, Dict]:
        """Decode machine assignment into full schedule.
        Returns (objective, tardiness, makespan, feasible, solution_dict)"""
        #--1.MACHINE ELIGIBILITY--
        jobs_by_machine = [[] for _ in range(self.n_machines + 1)]
        for job_id in range(self.n_jobs):
            machine_id = machine_assignment[job_id]
            if machine_id not in self.eligible[job_id]:
                return 10**9, 10**9, 10**9, False, {}
            jobs_by_machine[machine_id].append(job_id)
        
        start_times = [0] * self.n_jobs
        finish_times = [0] * self.n_jobs
        makespan = 0
        tardiness = 0
        

        for machine_id in range(1, self.n_machines + 1):
            machine_sequence = jobs_by_machine[machine_id]
            if not machine_sequence:
                continue
                
            machine_time = 0
            previous_job_id = None
            
            for job_id in machine_sequence:
                #--4.SETUP TIMES--
                if previous_job_id is None:
                    machine_time = max(machine_time, self.initial_setup[job_id])
                else:
                    machine_time += self.setup[job_id][previous_job_id]
                
                #--3.JOB PRECEDENCES--
                for predecessor_job_id in self.precedences[job_id]:
                    machine_time = max(machine_time, finish_times[predecessor_job_id - 1])
                
                start_times[job_id] = machine_time
                finish_times[job_id] = start_times[job_id] + self.processing[job_id]
                machine_time = finish_times[job_id]
                previous_job_id = job_id
                    


            #--2.NON-OVERLAPPING JOBS--
            
            
            #--5.KEEPING AVAILABLE RESOURCE CAPACITY--
            #--6.EXTRAS--
            #TODO: no detection of precedence cycles
            #TODO no resource management
            #TODO precedence not truly validated globally
            #TODO no explicit validation of machine id range
            makespan = max(makespan, machine_time)
        
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
