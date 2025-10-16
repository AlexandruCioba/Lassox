from experiment_db import ExperimentDB

db = ExperimentDB("lasso_experiments.db")
df = db.to_dataframe()

if df.empty:
    print("No experiments in database yet.")
else:
    print(f"Total experiments: {len(df)}")
    print(f"\nBy solver type:")
    print(df.groupby('solver_type').size())
    print(f"\nAverage test MSE by solver:")
    print(df.groupby('solver_type')['test_mse'].mean().round(6))
    print(f"\nAverage time by solver:")
    print(df.groupby('solver_type')['time'].mean().round(4))
    print(f"\nAverage time per iteration by solver (ms):")
    print((df.groupby('solver_type')['time_per_iter'].mean() * 1000).round(4))

db.close()

