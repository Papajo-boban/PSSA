import random
import copy
import math
import time
import statistics
from typing import List, Dict
from pms_instance import PMSInstance

class SimulatedAnnealing:
    def __init__(self, instance: PMSInstance, **kwargs):
        self.instance = instance
        self.initial_temp = kwargs.get('initial_temp', 5000.0)
        self.cooling_rate = kwargs.get('cooling_rate', 0.995)
        self.min_temp = kwargs.get('min_temp', 0.01)
        self.iter_per_temp = kwargs.get('iter_per_temp', 300)
        self.max_runtime = kwargs.get('max_runtime', 60)
        
    def random_solution(self) -> List[int]:
        """Generate random machine assignment"""
        assignment = []
        for j in range(self.instance.n_jobs):
            elig = list(self.instance.eligible[j])
            assignment.append(random.choice(elig))
        return assignment
    
    def get_neighbor(self, current: List[int]) -> List[int]:
        """Strong neighborhood: multiple move types"""
        neighbor = current[:]
        move = random.random()
        
        if move < 0.35:  
            # 1. Reassign job to different eligible machine
            j = random.randint(0, self.instance.n_jobs - 1)
            elig = list(self.instance.eligible[j])
            if len(elig) > 1:
                current_m = neighbor[j]
                new_m = random.choice([m for m in elig if m != current_m])
                neighbor[j] = new_m
                
        elif move < 0.70:
            # 2. Swap two jobs (anywhere)
            i, j = random.sample(range(self.instance.n_jobs), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        else:
            # 3. Swap two jobs on the same machine
            m = random.randint(1, self.instance.n_machines)
            jobs_on_m = [idx for idx, mach in enumerate(neighbor) if mach == m]
            if len(jobs_on_m) >= 2:
                i, j = random.sample(jobs_on_m, 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor
    
    def run(self) -> Dict:
        current = self.random_solution()
        current_obj, current_tard, current_makespan, _, _ = self.instance.decode(current)
        
        best = current[:]
        best_obj = current_obj
        best_tard = current_tard
        best_makespan = current_makespan
        
        temp = self.initial_temp
        start_time = time.time()
        iterations = 0
        
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
                        best = current[:]
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
            "runtime": round(time.time() - start_time, 2)
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
        "results": results
    }
    
    print(f"Best: {summary['best_objective']} | Avg: {summary['avg_objective']} ± {summary['std_objective']}")
    return summary
