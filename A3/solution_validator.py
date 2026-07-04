import sys
import json
from itertools import combinations
from enum import IntEnum

def get_sol_start_time(sol_data, job_id):
    return next((item["StartTime"] for item in sol_data["Jobs"] if item["JobId"] == job_id), -1)


def get_sol_machine_id(sol_data, job_id):
    return next((item["MachineId"] for item in sol_data["Jobs"] if item["JobId"] == job_id), -1)


def get_inst_job(instance_data, job_id):
    return next((item for item in instance_data["Jobs"] if item["Id"] == job_id), -1)


def check_machine_eligibility(instance_data, sol_data):
    for j in instance_data["Jobs"]:
        job_id = j["Id"]
        machine_id = get_sol_machine_id(sol_data, job_id)
        if machine_id == -1:
            print(f"Constraint violation: Job {job_id} has not been scheduled to any machine.")
            sol_data["Feasible"] = False
        elif machine_id not in j["EligibleMachineIds"]:
            print(f"Constraint violation: Job {job_id} has been scheduled to ineligible machine {machine_id}.")
            sol_data["Feasible"] = False


def check_no_job_overlap(instance_data, sol_data):
    for j1, j2 in combinations(instance_data["Jobs"], 2):
        j1_id, j2_id = j1["Id"], j2["Id"]
        j1_m, j2_m = get_sol_machine_id(sol_data, j1_id), get_sol_machine_id(sol_data, j2_id)

        if j1_id != j2_id and j1_m == j2_m:
            j1_start, j2_start = get_sol_start_time(sol_data, j1_id), get_sol_start_time(sol_data, j2_id)
            j1_end, j2_end = j1_start + j1["ProcessingTime"], j2_start + j2["ProcessingTime"]
            if j1_start < j2_end and j2_start < j1_end:
                print(f"Constraint violation: Jobs {j1_id} and {j2_id} are overlapping on machine {j1_m}.")
                sol_data["Feasible"] = False


def check_job_precedences(instance_data, sol_data):
    for j in instance_data["Jobs"]:
        j_id = j["Id"]
        for p in j["PrecedenceJobIds"]:
            end_p = get_sol_start_time(sol_data, p) + get_inst_job(instance_data, p)["ProcessingTime"]
            start_j = get_sol_start_time(sol_data, j_id)

            if end_p > start_j:
                print(f"Constraint violation: Precedence Job {p} is not finished before the start of Job {j_id}.")
                sol_data["Feasible"] = False


def check_setup_times(instance_data, sol_data):
    for m in instance_data["Machines"]:
        scheduled_jobs = sorted([j for j in sol_data["Jobs"] if j["MachineId"] == m["Id"]],
                                key=lambda k: k["StartTime"])

        if len(scheduled_jobs) == 0:
            continue

        j1 = scheduled_jobs[0]
        j1_id, j1_start = j1["JobId"], j1["StartTime"]
        j1_setup = get_inst_job(instance_data, j1_id)["InitialSetupTime"]
        if j1_start < j1_setup:
            print(f"Constraint violation: Job {j1_id} does not use enough initial setup time "
                  f"(actual: {j1_start}, required: {j1_setup}).")
            sol_data["Feasible"] = False

        for j in range(1, len(scheduled_jobs)):
            j1, j2 = scheduled_jobs[j - 1], scheduled_jobs[j]
            j1_id, j2_id = j1["JobId"], j2["JobId"]
            j1_start, j2_start = j1["StartTime"], j2["StartTime"]
            j1_end = j1_start + get_inst_job(instance_data, j1_id)["ProcessingTime"]
            j2_setup = get_inst_job(instance_data, j2_id)["JobSetupTimes"][j1_id - 1]
            if j2_start < j1_end + j2_setup:
                print(f"Constraint violation: Job {j2_id} does not have enough setup time after Job {j1_id}"
                      f"(actual: {j2_start - j1_end}, required: {j2_setup}).")
                sol_data["Feasible"] = False

class ResourceEvent(IntEnum):
    JOB_END = 0
    PERIOD_START = 1,
    JOB_START = 2

def check_resource_availabilities(instance_data, sol_data):
    events = []
    # Generate job events
    for j in sol_data["Jobs"]:
        j_id, j_start = j["JobId"], j["StartTime"]
        j = get_inst_job(instance_data, j_id)
        j_end = j_start + j["ProcessingTime"]
        for r in j["RequiredResources"]:
            r_id, c = r["ResourceId"], r["Capacity"]
            events.extend([
                {"JobId": j_id, "ResourceId": r_id, "Time": j_start, "Type": ResourceEvent.JOB_START, "Capacity": c},
                {"JobId": j_id, "ResourceId": r_id, "Time": j_end, "Type": ResourceEvent.JOB_END, "Capacity": c}
            ])
    # Generate Resource Events
    for r in instance_data["Resources"]:
        r_id = r["Id"]
        periods = r["AvailabilityPeriods"]
        for p in periods:
            events.append(
                {"ResourceId": r_id, "Time": p["Start"], "Type": ResourceEvent.PERIOD_START, "Capacity": p["Capacity"]}
            )
        events.append(
            {"ResourceId": r_id, "Time": periods[-1]["End"], "Type": ResourceEvent.PERIOD_START, "Capacity": 0}
        )
    events.sort(key=lambda k: (k["Time"], k["Type"]))

    # Go over all events and check resource availabilities
    cur_capacity = [0 for _ in instance_data["Resources"]]
    for e in events:
        r_id = e["ResourceId"]
        r_idx = r_id-1
        c = e["Capacity"]
        if e["Type"] == ResourceEvent.JOB_END:
            cur_capacity[r_idx] += c
        elif e["Type"] == ResourceEvent.PERIOD_START:
            cur_capacity[r_idx] = c
        elif e["Type"] == ResourceEvent.JOB_START:
            if cur_capacity[r_idx] < c:
                print(f"Constraint violation: There is not enough resource capacity for Job {e['JobId']}"
                      f" at time {e['Time']} for Resource {r_id} ({c - cur_capacity[r_idx]} capacity missing)")
                sol_data["Feasible"] = False
            cur_capacity[r_idx] -= c

def calculate_tardiness(instance_data, sol_data):
    tardiness = 0
    for j in instance_data["Jobs"]:
        end_time = get_sol_start_time(sol_data, j["Id"]) + j["ProcessingTime"]
        due_time = j["DueTime"]
        tardiness += max(0, end_time - due_time)

    return tardiness

def calculate_makespan(instance_data, sol_data):
    makespan = 0
    for j in instance_data["Jobs"]:
        end_time = get_sol_start_time(sol_data, j["Id"]) + j["ProcessingTime"]
        makespan = max(makespan, end_time)

    return makespan

def validate_solution(instance_filename, solution_filename):
    instance_data = dict()
    with open(instance_filename) as f:
        instance_data = json.load(f)
    sol_data = dict()
    with open(solution_filename) as f2:
        sol_data = json.load(f2)

    sol_data["Feasible"] = True

    check_machine_eligibility(instance_data, sol_data)

    check_no_job_overlap(instance_data, sol_data)

    check_job_precedences(instance_data, sol_data)

    check_setup_times(instance_data, sol_data)

    check_resource_availabilities(instance_data, sol_data)

    if sol_data["Feasible"]:
        print(f"Solution is feasible")
    else:
        print(f"There are constraint violations, the solution is infeasible")

    tardiness = calculate_tardiness(instance_data, sol_data)
    print(f"Tardiness: {tardiness}")

    makespan = calculate_makespan(instance_data, sol_data)
    print(f"Makespan: {makespan}")

    print(f"Total solution cost: {tardiness + makespan}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python solution_validator.py <instance_filename> <solution_filename>")
    else:
        instance_file = sys.argv[1]
        solution_file = sys.argv[2]
        validate_solution(instance_file, solution_file)
