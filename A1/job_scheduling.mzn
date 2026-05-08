
%include "conv_instances/PSSAI_PMS_j10_m3_r0_2.dzn";
%include "conv_instances/PSSAI_PMS_j10_m4_r1_4.dzn";
%include "conv_instances/PSSAI_PMS_j50_m1_r5_3.dzn";
%include "conv_instances/PSSAI_PMS_j100_m1_r13_2.dzn";
%include "500/PSSAI_PMS_j500_m3_r33_2.dzn";
int: n_jobs;
int: n_machines;
int: n_resources;
int: max_periods;

set of int: JOBS = 1..n_jobs;
set of int: MACHINES = 1..n_machines;
set of int: RESOURCES = 1..n_resources;


array[JOBS] of int: processing;
array[JOBS] of int: due;
array[JOBS] of set of MACHINES: eligible;

array[JOBS] of set of JOBS: precedences;
array[JOBS] of int: initial_setup;
array[JOBS] of array[JOBS] of int: setup;

int: base_horizon = sum(j in JOBS)(processing[j] + initial_setup[j]) + sum(j in JOBS)(max(i in JOBS where i != j)(setup[j][i]));
int: horizon = (3 * base_horizon) div 2;

array[JOBS] of var 0..horizon: start;
array[JOBS] of var 0..horizon: end;
array[JOBS] of var MACHINES: machines;
array[JOBS,JOBS] of var bool: before;

array[JOBS] of var 0..horizon: tardiness;
var 0..horizon: makespan;


array[RESOURCES] of int: n_res_periods;
array[JOBS] of array[RESOURCES] of int: required_resource;
array[RESOURCES] of array[1..max_periods] of int: res_period_start;
array[RESOURCES] of array[1..max_periods] of int: res_period_end;
array[RESOURCES] of array[1..max_periods] of int: res_period_capacity;



% --1.MACHINE ELIGIBILITY--
constraint forall(j in JOBS)(
  machines[j] in eligible[j]
);


% --2.NON-OVERLAPPING JOBS--
%defining before
constraint forall(i,j in JOBS where i!=j)(
  before[i,j] -> end[i] <= start[j]
);

array[JOBS] of var bool: is_first;

constraint forall (j in JOBS)(
  is_first[j] <-> forall (i in JOBS where i!=j)(
    (machines[i] != machines[j]) \/ not before[i,j]
  )
);

constraint forall(j in JOBS)(
  is_first[j] -> start[j] >= initial_setup[j]
);

constraint forall(i,j in JOBS where i<j)(
    (machines[i] != machines[j]) \/
    (before[i,j]) \/
    (before[j,i])
);

constraint forall(i,j in JOBS where i<j)(
    not (before[i,j] /\ before[j,i])
);


% --3.JOB PRECEDENCES--
constraint forall(j in JOBS, p in precedences[j])(
  end[p]<=start[j]
);



% --4.SETUP TIMES--

constraint forall(i,j in JOBS where i!=j)(
  before[i,j] /\ machines[i] = machines[j] -> 
  end[i] + setup[j][i] <= start[j]  
);

% --5.KEEPING AVAILABLE RESOURCE CAPACITY--
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


% --6.EXTRAS--
constraint forall(j in JOBS)(
    end[j] = start[j] + processing[j]
);


constraint forall(j in JOBS)(
    tardiness[j] >= end[j] - due[j] /\
    tardiness[j] >= 0
);

constraint makespan = max(j in JOBS)(end[j]);
%constraint forall(j in JOBS)(end[j] <= makespan);
%constraint forall(j in JOBS)(start[j] + processing[j] <= makespan);



solve minimize sum(j in JOBS)(tardiness[j]) + makespan;
%solve satisfy;



output [
  "{\n",
  "    \"Jobs\": [\n",
  concat([
    "        {\n" ++
    "            \"JobId\": " ++ show(j) ++ ",\n" ++
    "            \"StartTime\": " ++ show(start[j]) ++ ",\n" ++
    "            \"MachineId\": " ++ show(machines[j]) ++ "\n" ++
    "        }" ++
    if j < max(JOBS) then ",\n" else "\n" endif
    | j in JOBS
  ]),
  "    ]\n",
  "}\n"
];