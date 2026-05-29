"""
Baseline batch run for spatial repression-mobilization model.
540 runs: 3 env x 3 threshold_mean x 3 threshold_std x 2 obs_radius x 10 seeds
"""

import pandas as pd
from mesa.batchrunner import batch_run
from v3_model import RepressionModel

if __name__ == "__main__":
    params = {
        "env_type":             ["kampung", "mixed", "open"],
        "threshold_mean":       [0.25, 0.30, 0.35],
        "threshold_std":        [0.10, 0.15, 0.20],
        "observation_radius":   [0, 1],
        "n_agents":             [200],
        "repression_step":      [1],
        "instigator_threshold": [0.15],
        "dispersal_duration":   [3],
        "cell_capacity":        [15],
        "max_steps":            [100],
        "rng":                 list(range(10)),
    }

    total = 3 * 3 * 3 * 2 * 10
    print(f"Running {total} parameter combinations...")

    results = batch_run(
        RepressionModel,
        parameters=params,
        max_steps=100,
        data_collection_period=-1,
        display_progress=True,
    )

    df = pd.DataFrame(results)
    df.to_csv("results_baseline.csv", index=False)
    print(f"Done. {len(df)} rows written to results_baseline.csv")

    print("\nMean MobilizedProportion by env_type and observation_radius:")
    if "MobilizedProportion" in df.columns:
        print(
            df.groupby(["observation_radius", "env_type"])["MobilizedProportion"]
            .mean()
            .unstack()
            .round(3)
        )