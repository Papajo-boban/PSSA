# Parallel Machine Scheduling
## using Simulated Annealing

Names  
____________________________

Problem focus: Assign each job to an eligible machine while accounting for setup times, precedence constraints, tardiness, and overall completion time.

Runs per benchmark instance: 5  
Runtime cap per 100-job run: 45 s  
Objective: Tardiness + Makespan

---

# Simulated annealing alternates between random exploration and gradual tightening

The algorithm starts from a random assignment, evaluates each neighbor with `decode()`, and accepts worse moves with a temperature-controlled probability.

## Pseudocode

```text
procedure simulated_annealing
begin
  t <- 0
  initialize T
  select a current solution x_c at random
  evaluate x_c
  best <- x_c

  repeat
    repeat
      select a new solution x_n in the neighborhood of x_c
      evaluate x_n

      if x_n is infeasible then
        skip x_n
      else if eval(x_n) < eval(x_c) then
        x_c <- x_n
      else if random[0,1] < exp(-(eval(x_n) - eval(x_c)) / T) then
        x_c <- x_n

      if eval(x_c) < eval(best) then
        best <- x_c
    until (iterations at current temperature completed)

    T <- g(T, t)
    t <- t + 1
  until (T <= min_temp or runtime limit reached)

  return best
end
```

## Benefits

- Easy to implement with only a few tunable parameters.
- Escapes local optima by occasionally accepting worse moves early in the search.
- Scales better than exact methods on larger benchmark instances.
- Works well when the schedule evaluator is fast and deterministic.

---

# The project is split into instance loading, schedule decoding, and the annealing loop

Each file has a focused role, which keeps the experiment runner thin and makes the objective evaluation reusable.

## `pms_instance.py`

- Loads jobs, machines, setup times, and precedences from the JSON benchmark file.
- `decode()` converts a machine assignment into a concrete schedule.
- Returns objective, tardiness, makespan, feasibility, and a solution export.

## `pms_simulated_annealing.py`

- `random_solution()` builds the initial machine assignment.
- `get_neighbor()` applies reassignment and swap moves to perturb the current state.
- `run()` executes the annealing loop and tracks the best solution found.

## `run_experiments.py`

- Selects benchmark instances and annealing parameters.
- Runs each instance five times and prints best and average objectives.
- Writes solution JSON files and a combined `experiment_summary.json` summary.

Important implementation note: `decode()` evaluates assignments but does not explicitly optimize the within-machine sequence. That limits the search power on harder instances.

---

# The workflow is simple: choose instances, run the batch script, then inspect JSON outputs

The defaults in `run_experiments.py` already target the 100-job benchmark set with five repeated runs.

## Terminal commands

```bash
python run_experiments.py

# outputs
results/experiment_summary.json
results/*.solution.json
```

Adjust `INSTANCES` or SA parameters inside `run_experiments.py` when you want to test a different subset or runtime budget.

## Execution steps

1. Put benchmark JSON files in the `attachments/` folder.
2. Set the instance filenames listed in `INSTANCES`.
3. Run the script from the project root.
4. The solver performs 5 independent SA runs per instance.
5. Best solutions are saved as `.solution.json` files in `results/`.
6. The summary file stores best, average, and standard deviation values.

---

# Across five 100-job instances, the solver is stable and competitive on the multi-machine cases

Columns show the reference objective from `final_results.xlsx`, the best SA run, the average objective across 5 runs, and the standard deviation.

| Instance | Ref | SA Best | Avg x5 | Std Dev |
|---|---:|---:|---:|---:|
| `j100_m1_r13_2` | 238,580 | 238,580 | 238,580 | 0 |
| `j100_m2_r10_5` | 536,822 | 536,822 | 536,822 | 0 |
| `j100_m6_r15_1` | 69,085 | 68,777 | 71,580.2 | 3,305.15 |
| `j100_m7_r10_3` | 44,726 | 44,280 | 44,921 | 1,051.56 |
| `j100_m7_r18_4` | 64,748 | 62,930 | 64,657 | 1,417.34 |

Instances where SA best beat the reference table: 3 / 5  
Deterministic cases with zero variance: 2

- Two single-machine instances are deterministic, so every run converged to the same objective.
- On three multi-machine instances, SA found better objectives than the reference table stored in `final_results.xlsx`.

Source for the reference column: `final_results.xlsx` in the project root.

---

# The approach works, but the quality of the decoder and neighborhood matters as much as the annealing parameters

Most gains now depend on representing sequencing decisions more explicitly and enforcing all real constraints during the search.

## What worked

- The SA framework was easy to implement and tune in Python.
- Five repeated runs gave a useful view of stability, not just one lucky solution.
- The solver handled 100-job instances within a practical runtime budget.

## What should be improved next

- Model sequencing directly instead of only machine assignment, so setup-dependent ordering can really improve.
- Check resource constraints during the search, not only machine eligibility and precedence timing.
- Use richer neighborhoods or hybrid repair moves for larger and more constrained instances.
- Tune temperature schedules separately for small, medium, and large instance families.
