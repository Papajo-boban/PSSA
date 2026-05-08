import json
import sys
if len(sys.argv) != 3:
    print("Usage: python converter.py <instance.json> <output.dzn>")
    sys.exit(1)
with open(sys.argv[1]) as f:
    data = json.load(f)
n_jobs = len(data["Jobs"])
n_machines = len(data["Machines"])
n_resources = len(data.get("Resources", []))
max_time = 2000000
resources = data.get("Resources", [])
max_periods = max((len(r.get("AvailabilityPeriods", [])) for r in resources), default=0)
with open(sys.argv[2], "w") as f:
    print(f"n_jobs = {n_jobs};", file=f)
    print(f"n_machines = {n_machines};", file=f)
    print(f"n_resources = {n_resources};", file=f)
    print(f"max_periods = {max_periods};", file=f)
    # Basic data
    print("processing = [" + ",".join(str(j["ProcessingTime"]) for j in data["Jobs"]) + "];", file=f)
    print("due = [" + ",".join(str(j["DueTime"]) for j in data["Jobs"]) + "];", file=f)
    print("initial_setup = [" + ",".join(str(j.get("InitialSetupTime", 0)) for j in data["Jobs"]) + "];", file=f)
    print("eligible = [", end="", file=f)
    for j in data["Jobs"]:
        print("{" + ",".join(str(m) for m in j["EligibleMachineIds"]) + "},", end="", file=f)
    print("];", file=f)
    print("setup = [", end="", file=f)
    for j in data["Jobs"]:
        print("[" + ",".join(str(t) for t in j["JobSetupTimes"]) + "],", end="", file=f)
    print("];", file=f)
    print("precedences = [", end="", file=f)
    for j in data["Jobs"]:
        preds = j.get("PrecedenceJobIds", [])
        print("{" + ",".join(str(p) for p in preds) + "},", end="", file=f) if preds else print("{},", end="", file=f)
    print("];", file=f)
    # === RESOURCES - FIXED for both formats ===
    print("required_resource = [", end="", file=f)
    for j in data["Jobs"]:
        req = j.get("RequiredResources")
        if not req:
            row = [0] * n_resources
        elif isinstance(req[0], dict):          # format: [{"ResourceId":1, "Amount":5}, ...]
            amounts = [0] * n_resources
            for item in req:
                rid = item.get("ResourceId", 0) - 1
                if 0 <= rid < n_resources:
                    amounts[rid] = item.get("Capacity", item.get("Amount", 0))
            row = amounts
        else:                                         # format: [5, 0, 3, ...]
            row = [int(x) for x in req]
            if len(row) < n_resources:
                row += [0] * (n_resources - len(row))
            elif len(row) > n_resources:
                row = row[:n_resources]
        print("[" + ",".join(str(x) for x in row) + "],", end="", file=f)
    print("];", file=f)
    # Time-varying periods
    print("n_res_periods = [", end="", file=f)
    for r in data.get("Resources", []):
        print(len(r.get("AvailabilityPeriods", [])), end=",", file=f)
    print("];", file=f)
    print("res_period_start = [", end="", file=f)
    for r in data.get("Resources", []):
        periods = r.get("AvailabilityPeriods", [])
        row = [p.get("Start", p.get("StartTime", 0)) for p in periods] + [0] * (max_periods - len(periods))
        print("[" + ",".join(str(x) for x in row) + "],", end="", file=f)
    print("];", file=f)
    print("res_period_end = [", end="", file=f)
    for r in data.get("Resources", []):
        periods = r.get("AvailabilityPeriods", [])
        row = [p["End"] if "End" in p else p["EndTime"] for p in periods] + [0] * (max_periods - len(periods))
        print("[" + ",".join(str(x) for x in row) + "],", end="", file=f)
    print("];", file=f)
    print("res_period_capacity = [", end="", file=f)
    for r in data.get("Resources", []):
        periods = r.get("AvailabilityPeriods", [])
        row = [p.get("Capacity", 0) for p in periods] + [0] * (max_periods - len(periods))
        print("[" + ",".join(str(x) for x in row) + "],", end="", file=f)
    print("];", file=f)
print("✅ Conversion done – resources fully included!")
