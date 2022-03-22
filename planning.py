import time

def abstract_plan(task, hlc):
    start = time.time()
    for high_level_plan, budget_percent in hlc(task):
        start2 = time.time()
        while (time.time() - start2) < budget_percent * time_budget:



