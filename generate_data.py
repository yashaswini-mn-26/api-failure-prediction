import pandas as pd
import random

data = []

for _ in range(5000):
    response_time = random.randint(50, 2000)
    status_code = random.choice([200, 200, 200, 500, 503])
    cpu = random.randint(10, 100)
    memory = random.randint(20, 100)

    failure = 1 if (response_time > 1000 or status_code >= 500 or cpu > 85) else 0

    data.append([response_time, status_code, cpu, memory, failure])

df = pd.DataFrame(data, columns=[
    "response_time", "status_code", "cpu_usage", "memory_usage", "failure"
])

df.to_csv("api_logs.csv", index=False)

print("Dataset created ✅")