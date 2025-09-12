import pandas as pd
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

basins = pd.read_table(
    "/home/aldotapia/data/camels/camels_name.txt", sep=";", dtype="str"
)
basins = basins["gauge_id"].tolist()

basins = basins[:150]

print(f"Starting calibration for {len(basins)} basins.")


def run_calibration(basin):
    cmd = [
        "python",
        "/home/aldotapia/sacsma_aldo/calibrate_single_basin.py",
        "--basin",
        basin,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_calibration, basin) for basin in basins]
        for future in as_completed(futures):
            print(future.result())
