import pandas as pd

data = pd.read_csv("data/summary/full_scale_1.csv")
print(pd.to_timedelta(data["run_time"]).dt.total_seconds().mean())
