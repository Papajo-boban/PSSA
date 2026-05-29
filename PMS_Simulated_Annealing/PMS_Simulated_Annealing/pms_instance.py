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
        
        self.eligible: List[Set[int]] = [set(j["EligibleMachineIds"]) for j in data["Jobs"]]
        self.precedences: List[Set[int]] = [set(j["PrecedenceJobIds"]) for j in data["Jobs"]]
        
        # setup[j][k] = setup time when job j follows job k (0-based)
        self.setup = [j["JobSetupTimes"] for j in data["Jobs"]]
        
        self.resources = data.get("Resources", [])
        self.required_resources = []
        for j in data["Jobs"]:
            req = [0] * max(1, self.n_resources)
            for r in j.get("RequiredResources", []):
                if r["ResourceId"] <= len(req):
                    req[r["ResourceId"]-1] = r["Capacity"]
            self.required_resources.append(req)

    def decode(self, machine_assignment: List[int]) -> Tuple[int, int, int, bool, Dict]:
        """Decode machine assignment into full schedule.
        Returns (objective, tardiness, makespan, feasible, solution_dict)"""
        machine_jobs = [[] for _ in range(self.n_machines + 1)]
        for j in range(self.n_jobs):
            m = machine_assignment[j]
            if m not in self.eligible[j]:
                return 10**9, 10**9, 10**9, False, None
            machine_jobs[m].append(j)
        
        start = [0] * self.n_jobs
        finish = [0] * self.n_jobs
        makespan = 0
        tardiness = 0
        
        for m in range(1, self.n_machines + 1):
            seq = machine_jobs[m]
            if not seq:
                continue
                
            current_time = 0
            prev = None
            
            for job_idx in seq:
                # Setup time
                if prev is None:
                    current_time = max(current_time, self.initial_setup[job_idx])
                else:
                    current_time += self.setup[job_idx][prev]
                
                # Precedence constraints
                for p in self.precedences[job_idx]:
                    current_time = max(current_time, finish[p - 1])
                
                start[job_idx] = current_time
                finish[job_idx] = current_time + self.processing[job_idx]
                current_time = finish[job_idx]
                prev = job_idx
            
            makespan = max(makespan, current_time)
        
        tardiness = sum(max(0, finish[j] - self.due[j]) for j in range(self.n_jobs))
        obj = tardiness + makespan
        
        solution = {
            "Jobs": [
                {
                    "JobId": j+1, 
                    "StartTime": int(start[j]), 
                    "MachineId": int(machine_assignment[j])
                }
                for j in range(self.n_jobs)
            ]
        }
        
        return obj, tardiness, makespan, True, solution

    def is_feasible(self, machine_assignment: List[int]) -> bool:
        _, _, _, feasible, _ = self.decode(machine_assignment)
        return feasible
