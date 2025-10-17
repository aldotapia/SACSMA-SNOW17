import pandas as pd
import src.utilities.camels_utilities as camels
import matplotlib.pyplot as plt
import numpy as np
from src import metrics as m
import argparse
import spotpy
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--basin", type=str, help="Basin to calibrate")
argparser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility"
)

args = argparser.parse_args()
basin = args.basin
seed = args.seed

np.random.seed(seed)

base_path = "/home/aldotapia/data/new_cal/"
save_path = "/home/aldotapia/data/new_cal/new_files"

# if not os.path.exists(f'{base_path}/{seed}'):
#    os.makedirs(f'{base_path}/{seed}')


from src.optimization import get_initial_state, Spotpy_setup

print(f"Calibrating basin : {basin}")
try:
    forcings, area, lat, elev = camels.load_forcings(basin)
    benchmark = camels.load_discharge(basin)
    params = camels.load_sacsma_parameters(basin)
    usgs = camels.load_usgs(basin, area)
except:
    print(f"Error loading data for basin {basin}. Skipping to next basin.")
    pd.Series({"basin": basin, "error": "data loading error"}).to_csv(
        f"{base_path}/{seed}/error_{basin}_{seed}.csv"
    )

adcs = {
    "adc1": params["adc1"],
    "adc2": params["adc2"],
    "adc3": params["adc3"],
    "adc4": params["adc4"],
    "adc5": params["adc5"],
    "adc6": params["adc6"],
    "adc7": params["adc7"],
    "adc8": params["adc8"],
    "adc9": params["adc9"],
    "adc10": params["adc10"],
    "adc11": params["adc11"],
}


forcings_1980 = forcings[
    (forcings.index >= "1980-10-01") & (forcings.index < "1981-10-01")
]

initial_states, initial_cs = get_initial_state(forcings_1980, params, adcs, lat, elev)
ini_states_dict = {
    "basin": basin,
    "uztwc": initial_states[0],
    "uzfwc": initial_states[1],
    "lztwc": initial_states[2],
    "lzfsc": initial_states[3],
    "lzfpc": initial_states[4],
    "adimc": initial_states[5],
}

pd.Series(ini_states_dict).to_csv(
    f"{base_path}/{seed}/initial_states_{basin}_{seed}.csv"
)

forcings = forcings.merge(usgs["QObs"], left_index=True, right_index=True)
forcings.loc[forcings["QObs"] < 0, "QObs"] = np.nan
forcings = forcings.dropna(subset=["QObs"])

mask_dates_eval = (forcings.index >= "1995-10-01") & (forcings.index < "2010-10-01")
mask_dates_train = (forcings.index >= "1981-10-01") & (forcings.index < "1995-10-01")

len_out = forcings.index < "1995-10-01"

sp = Spotpy_setup(
    forcings=forcings[len_out],
    observations=forcings["QObs"][len_out],
    latitude=lat,
    elevation=elev,
    params=params,
    adcs=adcs,
    initial_states=initial_states,
    algorithm_minimize=True,
    mask_dates=mask_dates_train[len_out],
    path=save_path,
    basin=basin,
    seed=seed,
)

sampler = spotpy.algorithms.sceua(
    sp,
    dbname=f"{base_path}/{seed}/all_params_{basin}_{seed}",
    dbformat="csv",
    save_sim=False,
)

max_model_runs = 100000

sampler.sample(
    max_model_runs, ngs=20, pcento=0.00000001, peps=0.00000001
)  # , max_loop_inc = 2)

all_params = pd.read_csv(f"{save_path}/all_params_{basin}_{seed}.csv")
rmse_values = pd.read_csv(f"{save_path}/rmse_{basin}_{seed}.csv")

min_rmse_index = rmse_values["rmse"].idxmin()
best_params = list(all_params.iloc[min_rmse_index])

sp = Spotpy_setup(
    forcings=forcings,
    observations=forcings["QObs"],
    latitude=lat,
    elevation=elev,
    params=params,
    adcs=adcs,
    initial_states=initial_states,
    algorithm_minimize=True,
    mask_dates=mask_dates_eval,
)

sim_flow = sp.simulation(best_params)

new_params = sp.params

nse_val = m.nse(forcings["QObs"][mask_dates_eval], sim_flow[mask_dates_eval])

new_params["nse"] = nse_val
new_params.to_csv(f"{base_path}/{seed}/best_params_{basin}_{seed}.csv")

# plot best params

plt.rcParams["figure.figsize"] = [10, 5]
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax2.bar(
    forcings.index[mask_dates_eval],
    forcings["prcp(mm/day)"][mask_dates_eval],
    color="blue",
    label="Precipitation",
    zorder=-0,
)
ax2.set_ylabel("$P\\;[mm]$")
ax2.set_ylim(0, forcings["prcp(mm/day)"][mask_dates_eval].max() * 2)
ax2.invert_yaxis()
ax1.scatter(
    usgs.index[mask_dates_eval],
    usgs["QObs"][mask_dates_eval],
    color="red",
    label="$Q_{obs}$",
    s=10,
)
ax1.plot(
    sim_flow.index[mask_dates_eval],
    sim_flow[mask_dates_eval],
    color="black",
    label="$Q_{sim}$",
    linewidth=1,
)
ax1.set_ylabel("$Q\\;[mm]$")
ax1.grid(True)
ax1.set_ylim(0, usgs["QObs"][mask_dates_eval].max() * 2)
ax1.legend(loc=0)
ax2.legend(loc=0)
ax1.set_xlabel("Time [days]")
ax1.set_title(f"Basin {basin} - NSE: {nse_val:.3f}, Seed: {seed}")
plt.savefig(f"{base_path}/{seed}/plot_{basin}_{seed}.pdf", bbox_inches="tight")
plt.show()
