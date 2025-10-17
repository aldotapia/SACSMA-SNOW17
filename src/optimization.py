"""Module for optimization setup using SPOTPY with the SAC-SMA + Snow17 model."""
import os
import pandas as pd
import numpy as np
from src.model import Sacsma

from spotpy.parameter import Uniform, Exponential
from spotpy.objectivefunctions import rmse


def get_initial_state(forcings, params, adcs, lat, elev):
    """Function to compute the initial states for the SAC-SMA model given forcings and parameters."""

    initial = np.array(
        [120000.0, 120000.0, 120000.0, 120000.0, 120000.0, 120000.0], dtype="f4"
    )  # uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc
    initial_cs = np.full(19, 0.0)

    flag = True
    i = 0

    while flag:
        # print(f'Iteration {i}')

        test = Sacsma()

        test.set_parameters(params)
        test.set_snow17_parameters(params)
        test.set_hydrograph_parameters(params)
        test.set_adc_parameters(pd.Series(adcs))
        test.set_forcings(
            date=forcings.index.to_series(),
            precip=forcings["prcp(mm/day)"],
            max_temp=forcings["tmax(C)"],
            min_temp=forcings["tmin(C)"],
            srad=forcings["srad(W/m2)"],
            dayl=forcings["dayl(s)"],
        )
        test.general_parameters.elevation = elev
        test.general_parameters.latitude = lat
        test.compute_et()
        test.set_initial_states(
            cs=initial_cs.astype("f4"),
            tprev=np.array(0.0),
            uztwc=initial[0],
            uzfwc=initial[1],
            lztwc=initial[2],
            lzfsc=initial[3],
            lzfpc=initial[4],
            adimc=initial[5],
        )

        test.run()
        end = [
            test.states.sacsma_uztwc[-1],
            test.states.sacsma_uzfwc[-1],
            test.states.sacsma_lztwc[-1],
            test.states.sacsma_lzfsc[-1],
            test.states.sacsma_lzfpc[-1],
            test.states.sacsma_adimc[-1],
        ]

        end_cs = test.last_cs.copy()

        if np.allclose(end, initial, rtol=1e-5) & np.allclose(
            end_cs, initial_cs, rtol=1e-5
        ):
            flag = False
        else:
            initial = np.array(end, dtype="f4")
            initial_cs = end_cs.copy()
            i += 1
    return end, initial_cs


class Spotpy_setup(object):
    """Class to setup SPOTPY optimization for SAC-SMA + Snow17 model."""
    uztwm = Uniform(low=1.0, high=800.0)
    uzfwm = Uniform(low=1.0, high=800.0)
    uzk = Uniform(low=0.1, high=0.7)
    zperc = Uniform(low=0.1, high=250.0)
    rexp = Uniform(low=0.0, high=6.0)
    lztwm = Uniform(low=1.0, high=800.0)
    lzfsm = Uniform(low=1.0, high=1000.0)
    lzfpm = Uniform(low=1.0, high=1000.0)
    lzsk = Exponential(minbound=0.001, maxbound=0.25, scale=1)
    lzpk = Exponential(minbound=0.00001, maxbound=0.025, scale=1)
    scf = Uniform(low=0.1, high=5.0)
    mfmax = Uniform(low=0.8, high=3.0)
    mfmin = Uniform(low=0.01, high=0.79)
    uadj = Uniform(low=0.01, high=0.40)
    si = Uniform(low=1.0, high=3500.0)
    tipm = Uniform(low=0.05, high=0.2)
    pxtemp = Uniform(low=-1.0, high=3.0)
    plwhc = Uniform(low=0.03, high=0.250)
    unit_shape = Uniform(low=1.0, high=5.0)
    unit_scale = Uniform(low=0.001, high=150.0)
    pet_coef = Uniform(low=1.26, high=1.74)

    def __init__(
        self,
        forcings: pd.DataFrame,
        observations: pd.Series,
        latitude: float,
        elevation: float,
        params: dict,
        adcs: dict,
        initial_states: np.ndarray,
        algorithm_minimize: bool,
        mask_dates: pd.Series,
        path: str = None,
        basin: str = None,
        seed: int = None,
    ):

        self.lat = latitude
        self.elev = elevation
        self.params = params
        self.initial_states = np.array(initial_states, dtype="f4")
        self.mask_dates = mask_dates
        self.forcings = forcings
        self.observations = observations
        self.algorithm_minimize = algorithm_minimize
        self.adcs = adcs
        self.path = path
        self.basin = basin
        self.seed = seed

    def simulation(self, x):
        self.params["uztwm"] = x[0]
        self.params["uzfwm"] = x[1]
        self.params["uzk"] = x[2]
        self.params["zperc"] = x[3]
        self.params["rexp"] = x[4]
        self.params["lztwm"] = x[5]
        self.params["lzfsm"] = x[6]
        self.params["lzfpm"] = x[7]
        self.params["lzsk"] = x[8]
        self.params["lzpk"] = x[9]
        self.params["scf"] = x[10]
        self.params["mfmax"] = x[11]
        self.params["mfmin"] = x[12]
        self.params["uadj"] = x[13]
        self.params["si"] = x[14]
        self.params["tipm"] = x[15]
        self.params["pxtemp"] = x[16]
        self.params["plwhc"] = x[17]
        self.params["unit_shape"] = x[18]
        self.params["unit_scale"] = x[19]
        self.params["pet_coef"] = x[20]

        model = Sacsma()
        model.set_parameters(self.params)
        model.set_snow17_parameters(self.params)
        model.set_hydrograph_parameters(self.params)
        model.set_adc_parameters(pd.Series(self.adcs))
        model.set_forcings(
            date=self.forcings.index.to_series(),
            precip=self.forcings["prcp(mm/day)"],
            max_temp=self.forcings["tmax(C)"],
            min_temp=self.forcings["tmin(C)"],
            srad=self.forcings["srad(W/m2)"],
            dayl=self.forcings["dayl(s)"],
        )
        model.general_parameters.elevation = self.elev
        model.general_parameters.latitude = self.lat
        model.compute_et(alpha_pt=self.params["pet_coef"])
        model.set_initial_states(
            cs=np.full(19, 0.0, dtype="f4"),
            tprev=np.array(0.0),
            uztwc=self.initial_states[0],
            uzfwc=self.initial_states[1],
            lztwc=self.initial_states[2],
            lzfsc=self.initial_states[3],
            lzfpc=self.initial_states[4],
            adimc=self.initial_states[5],
        )

        model.run()

        if self.basin != None:
            # since spotpy doesn't save all the simulation
            # results (no idea why), this is my workaround
            params = (
                list(model.parameters.__dict__.keys())
                + list(model.snow17_parameters.__dict__.keys())
                + list(model.hydrograph_parameters.__dict__.keys())
                + ["pet_coef", "lat", "elev"]
            )
            values = (
                list(model.parameters.__dict__.values())
                + list(model.snow17_parameters.__dict__.values())
                + list(model.hydrograph_parameters.__dict__.values())
                + list(model.general_parameters.__dict__.values())
            )
            obj_params = [
                "uztwm",
                "uzfwm",
                "uzk",
                "zperc",
                "rexp",
                "lztwm",
                "lzfsm",
                "lzfpm",
                "lzsk",
                "lzpk",
                "scf",
                "mfmax",
                "mfmin",
                "uadj",
                "si",
                "tipm",
                "pxtemp",
                "plwhc",
                "unit_shape",
                "unit_scale",
                "pet_coef",
            ]
            params_series = pd.Series(values, index=params)

            file = f"{self.path}/all_params_{self.basin}_{self.seed}.csv"

            if not os.path.exists(file):
                with open(file, "w") as f:
                    f.write((",".join(obj_params) + "\n"))
            with open(file, "a") as f:
                f.write(
                    ",".join([str(v) for v in list(params_series.loc[obj_params])])
                    + "\n"
                )

        return model.fluxes_df["sacsma_uh_qq"]

    def evaluation(self):
        return self.observations

    def objectivefunction(self, simulation, evaluation):
        obs = evaluation[self.mask_dates].values
        sim = simulation[self.mask_dates].values
        mask = ~np.isnan(obs)
        objectivefunction = rmse(obs[mask], sim[mask])
        if not self.algorithm_minimize:
            objectivefunction = -objectivefunction

        if self.basin != None:
            # save again for the workaround
            file = f"{self.path}/rmse_{self.basin}_{self.seed}.csv"
            if not os.path.exists(file):
                with open(file, "w") as f:
                    f.write("rmse\n")
            with open(file, "a") as f:
                f.write(f"{objectivefunction}\n")

        return objectivefunction
