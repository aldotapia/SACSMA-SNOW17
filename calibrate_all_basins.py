import pandas as pd
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob

import glob

basins_ = pd.read_table(
    "/home/aldotapia/data/camels/camels_name.txt", sep=";", dtype="str"
)
basins_ = basins_["gauge_id"].tolist()

base_path = "/home/aldotapia/data/new_cal/"

basins_done = []

for basin in basins_:

    filenames = glob.glob(f"{base_path}/**/best_params_{basin}_*.csv")
    filenames = sorted(filenames)

    if len(filenames) == 10:
        basins_done.append(basin)

basins_ = [val for val in basins_ if val not in basins_done]

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(f"Starting calibration for {len(basins_)} basins.")


def run_calibration(basin, seed):
    cmd = [
        "python",
        "/home/aldotapia/sacsma_aldo/calibrate_single_basin.py",
        "--basin",
        basin,
        "--seed",
        str(seed),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_calibration, basin, seed)
            for basin in basins_
            for seed in seeds
        ]
        for future in as_completed(futures):
            print(future.result())
