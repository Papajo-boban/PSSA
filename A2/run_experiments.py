#!/usr/bin/env python3
"""
run_experiments.py
Runs Simulated Annealing on multiple PSSAI_PMS benchmark instances.
Generates results suitable for Assignment 2 slides.
"""

import json
import os
from pms_instance import PMSInstance
from pms_simulated_annealing import run_multiple_times

# List of instances to run (you can add more)
INSTANCES = [
    "PSSAI_PMS_j10_m3_r0_1.json",
    "PSSAI_PMS_j10_m3_r0_2.json",
    "PSSAI_PMS_j10_m3_r1_5.json",
    "PSSAI_PMS_j10_m3_r2_3.json",
    "PSSAI_PMS_j10_m4_r1_4.json",
    "PSSAI_PMS_j50_m1_r5_3.json",
    "PSSAI_PMS_j50_m2_r8_1.json",
    "PSSAI_PMS_j50_m5_r5_5.json",
    "PSSAI_PMS_j50_m5_r8_2.json",
    "PSSAI_PMS_j50_m5_r8_4.json",
    "PSSAI_PMS_j100_m1_r13_2.json",
    "PSSAI_PMS_j100_m2_r10_5.json",
    "PSSAI_PMS_j100_m6_r15_1.json",
    "PSSAI_PMS_j100_m7_r10_3.json",
    "PSSAI_PMS_j100_m7_r18_4.json",
    "PSSAI_PMS_j500_m3_r33_2.json",
    "PSSAI_PMS_j500_m5_r31_5.json",
    # "PSSAI_PMS_j500_m7_r14_3.json",
    # "PSSAI_PMS_j500_m7_r25_4.json",
    # "PSSAI_PMS_j500_m8_r29_1.json"
]

def main():
    all_results = {}
    base_path = "attachments"
    
    os.makedirs("results", exist_ok=True)
    
    for inst_name in INSTANCES:
        full_path = os.path.join(base_path, inst_name)
        
        if not os.path.exists(full_path):
            print(f"Warning: {inst_name} not found. Skipping.")
            continue
        
        # Parameters tuned for small/medium instances
        summary = run_multiple_times(
            full_path,
            n_runs=5,
            initial_temp=8000.0,
            cooling_rate=0.993,
            iter_per_temp=200,
            max_runtime=120   # seconds per run
        )
        
        all_results[inst_name] = summary
        
        # Save best solution
        best_result = min(summary["results"], key=lambda x: x["objective"])
        sol_path = f"results/{inst_name.replace('.json', '.solution.json')}"
        
        with open(sol_path, 'w') as f:
            json.dump(best_result["solution"], f, indent=2)
        
        print(f"Saved best solution to {sol_path}\n")
    
    # Save overall summary
    with open("results/experiment_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("=== All experiments completed! ===")
    print("Check 'results/' folder for solutions and summary.")

if __name__ == "__main__":
    main()
