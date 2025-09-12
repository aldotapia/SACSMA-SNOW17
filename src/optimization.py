import pandas as pd
import numpy as np
from spotpy.parameter import Uniform, Exponential
from spotpy.objectivefunctions import rmse
from src.model import Sacsma


def get_initial_state(forcings, params, adcs, lat, elev):
    """
    Iteratively computes the initial hydrologic and snow model states that
    reach equilibrium for the given input data. The function initializes the
    model states and runs the Sacsma model repeatedly, updating the initial
    states each time, until the final states at the end of the simulation are
    sufficiently close to the initial states (within a relative tolerance).
    This ensures that the model starts from a stable state for the given
    forcings and parameters.

    Parameters
    ----------
        forcings : pd.DataFrame
            DataFrame containing meteorological forcing data with columns such
            as 'prcp(mm/day)', 'tmax(C)', 'tmin(C)', 'srad(W/m2)', 
            and 'dayl(s)'.
        params : dict or pd.Series
            Model parameters for Sacsma, Snow17, and hydrograph components.
        adcs : array-like
            Additional ADC parameters required by the model.
        lat : float
            Latitude of the catchment area.
        elev : float
            Elevation of the catchment area.

    Returns
    -------
        tuple:
            - end (list): Final equilibrium values for the main hydrologic 
              states [uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc].
            - initial_cs (np.ndarray): Final equilibrium values for the snow 
              model state variables (CS).
    """

    initial = np.array(
        [120000.0, 120000.0, 120000.0, 120000.0, 120000.0, 120000.0],
        dtype="f4"
    )  # uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc
    initial_cs = np.full(19, 0.0)

    flag = True
    i = 0

    while flag:
        print(f"Iteration {i}")

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

        print(
            f"UZTWC: {end[0]:.2f}, UZFWC: {end[1]:.2f}, LZTWC: {end[2]:.2f}, "
            f"LZFSC: {end[3]:.2f}, LZFPC: {end[4]:.2f}, ADIMC: {end[5]:.2f}"
        )
        print(f"CS: {initial_cs}")

        if np.allclose(end, initial, rtol=1e-5) & np.allclose(
            end_cs, initial_cs, rtol=1e-5
        ):
            flag = False
        else:
            initial = np.array(end, dtype="f4")
            initial_cs = end_cs.copy()
            i += 1
    return end, initial_cs


sacsma_parameter_keys = [
    "uztwm",  # Upper zone free water capacity [mm] [1.0-800.0]
    "uzfwm",  # Upper zone free water capacity [mm] [1.0-800.0]
    "uzk",  # Fractional daily upper zone free water withdrawal rate [/day] [0.1–0.7]
    "pctim",  # Minimum impervious area [%]
    "adimp",  # Additional impervious are [%]
    "riva",  # Riparian vegetation area [%]
    "zperc",  # Percolation potential - maximum percolation rate [dimensionless] [1.0-250.0]
    "rexp",  # Exponent for the percolation equation [0.0-6.0]
    "lztwm",  # Lower zone tension water capacity [mm] [1.0-800.0]
    "lzfsm",  # Lower zone supplemental free water capacity [mm] [1.0-1000.0]
    "lzfpm",  # Lower zone primary free water capacity [mm] [1.0-1000.0]
    "lzsk",  # Fractional daily supplemental withdrawal rate [/day] [0.001–0.25]
    "lzpk",  # Fractional daily primary withdrawal rate [/day] [0.00001–0.025]
    "pfree",  # % of percolated water which always goes directly to lower zone free water storages
    "side",  # Ratio of non-channel baseflow (deep recharge) to channel (visible) baseflow
    "rserv",  # % of lower zone free water which cannot be transferred to lower zone tension water
]
snow17_parameter_keys = [
    "scf",  # Snowe correction factor [dimensionless] [0.1-5.0]
    "mfmax",  # Maximum snowmelt factor [mm/°C] [0.8-3.0]
    "mfmin",  # Minimum snowmelt factor [mm/°C] [0.01-0.79]
    "uadj",  # 1-m wind speed adjustment factor [km/6h] low in covered areas, high in open areas [0.01-0.40]
    "si",  # Mean areal water equivalent above which the area essentially has 100% snow cover [mm] [1.0-3500.0]
    "nmf",  # Maximum negative melt factor [mm/°C/6h]
    "tipm",  # Antecedent temperature index parameter [dimensionless], range from 0-1
    "mbase",  # Base temperature for snowmelt computations during non-rain periods [°C]
    "pxtemp",  # Temperature that separates rain from snow [°C] [-1.0-3.0]
    "plwhc",  # Percent liquid water holding capacity [dimensionless], range from 0-1
    "daygm",  # Constant daily amount of melt which takes place at the snow-soil interface whenever there is a snow cover [mm/day]
]
hydrograph_parameter_keys = [
    "unit_shape",  # Shape of unit hydrograph [1.0-5.0]
    "unit_scale",  # Scale of unit hydrograph [0.001-150.0]
]

state_keys = [
    "sacsma_snow17_tprev",
    "sacsma_uztwc",  # Upper zone tension water content [mm]
    "sacsma_uzfwc",  # Upper zone free water content [mm]
    "sacsma_lztwc",  # Lower zone tension water content [mm]
    "sacsma_lzfsc",  # Lower zone free supplemental content [mm]
    "sacsma_lzfpc",  # Lower zone free primary content [mm]
    "sacsma_adimc",  # Tension water content of the ADIMP area [mm] - If not known, use UZTWC+LZTWC
]
flux_keys = [
    "sacsma_pet",  # Potential evapotranspiration [mm/day]
    "sacsma_snow17_raim",  # Rain plus melt
    "sacsma_snow17_sneqv",  # Snow water equivalent [mm]
    "sacsma_snow17_snow",  # Snowfall [mm]
    "sacsma_snow17_snowh",  # Snow depth [mm]
    "sacsma_surf",  # Impervious area runoff + Direct runoff + Surface runoff + Interflow
    "sacsma_grnd",  # Non-channel baseflow + some kind of baseflow
    "sacsma_qq",  # Total channel inflow
    "sacsma_tet",  # Total evapotranspiration
]


class SpotpySetup(object):
    """
    SpotpySetup is a parameter optimization interface for the SACSMA-SNOW17
    hydrological model.

    Attributes
    ----------
        uztwm : Uniform)
            Upper zone free water capacity [mm]
        uzfwm : Uniform)
            Upper zone free water capacity [mm]
        uzk : Uniform)
            Fractional daily upper zone free water withdrawal rate [/day]
        zperc : Uniform)
            Percolation potential - maximum percolation rate [dimensionless]
        rexp : Uniform)
            Exponent for the percolation equation
        lztwm : Uniform)
            Lower zone tension water capacity [mm]
        lzfsm : Uniform)
            Lower zone supplemental free water capacity [mm]
        lzfpm : Uniform)
            Lower zone primary free water capacity [mm]
        lzsk : Exponential)
            Fractional daily supplemental withdrawal rate [/day]
        lzpk : Exponential)
            Fractional daily primary withdrawal rate [/day]
        scf : Uniform)
            Snow correction factor [dimensionless]
        mfmax : Uniform)
            Maximum snowmelt factor [mm/°C]
        mfmin : Uniform)
            Minimum snowmelt factor [mm/°C]
        uadj : Uniform)
            1-m wind speed adjustment factor [km/6h]
        si : Uniform)
            Mean areal water equivalent above which the area essentially has
            100% snow cover [mm]
        tipm : Uniform)
            Antecedent temperature index parameter [dimensionless]
        pxtemp : Uniform)
            Temperature that separates rain from snow [°C]
        plwhc : Uniform)
            Percent liquid water holding capacity [dimensionless]
        unit_shape : Uniform)
            Shape parameter for unit hydrograph [dimensionless]
        unit_scale : Uniform)
            Scale parameter for unit hydrograph [dimensionless]
        pet_coef : Uniform)
            Potential evapotranspiration coefficient [mm/day]

    Parameters
    ----------
        forcings : pd.DataFrame
            Meteorological input data indexed by date.
        observations : pd.Series)
            Observed streamflow or target variable.
        latitude : float
            Latitude of the catchment.
        elevation : float
            Elevation of the catchment.
        params : dict
            Initial model parameters.
        adcs : dict
            Additional distributed catchment parameters.
        initial_states : np.ndarray
            Initial state variables for the model.
        algorithm_minimize : bool
            If True, objective function is minimized; if False, maximized.
        mask_dates : pd.Series
            Boolean mask for valid dates in evaluation.

    Methods
    -------
        simulation(x):
            Runs the SACSMA-SNOW17 model with parameter vector x and returns
            simulated streamflow.
        evaluation():
            Returns the observed data for evaluation.
        objectivefunction(simulation, evaluation):
            Calculates the RMSE between simulation and evaluation, applying
            mask and sign convention.
    """

    uztwm = Uniform(low=1.0, high=800.0, step=0.001)
    uzfwm = Uniform(low=1.0, high=800.0, step=0.001)
    uzk = Uniform(low=0.1, high=0.7, step=0.001)
    zperc = Uniform(low=0.1, high=250.0, step=0.001)
    rexp = Uniform(low=0.0, high=3.0, step=0.001)
    lztwm = Uniform(low=1.0, high=800.0, step=0.001)
    lzfsm = Uniform(low=1.0, high=1000.0, step=0.001)
    lzfpm = Uniform(low=1.0, high=1000.0, step=0.001)
    lzsk = Exponential(minbound=0.001, maxbound=0.25, step=0.001, scale=1)
    lzpk = Exponential(minbound=0.00001, maxbound=0.25, step=0.00001, scale=1)
    scf = Uniform(low=0.1, high=5.0, step=0.001)
    mfmax = Uniform(low=0.8, high=3.0, step=0.001)
    mfmin = Uniform(low=0.01, high=0.79, step=0.001)
    uadj = Uniform(low=0.01, high=0.40, step=0.001)
    si = Uniform(low=1.0, high=3500.0, step=0.001)
    tipm = Uniform(low=0.0, high=1.0, step=0.001)
    pxtemp = Uniform(low=-1.0, high=3.0, step=0.001)
    plwhc = Uniform(low=0.0, high=1.0, step=0.001)
    unit_shape = Uniform(low=1.0, high=5.0, step=0.001)
    unit_scale = Uniform(low=0.001, high=150.0, step=0.001)
    pet_coef = Uniform(low=1.26, high=1.74, step=0.001)

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

    def simulation(self, x):
        """
        Runs a hydrological simulation using the SACSMA-SNOW17 model with the
        provided parameter vector.

        Parameters
        ----------
        x : array-like
            A sequence of model parameters in the following order:
            [uztwm, uzfwm, uzk, zperc, rexp, lztwm, lzfsm, lzfpm, lzsk, lzpk,
             scf, mfmax, mfmin, uadj, si, tipm, pxtemp, plwhc, unit_shape,
             unit_scale, pet_coef].

        Returns
        -------
        pandas.Series
            Simulated streamflow values ("sacsma_uh_qq") from the model run.
        """
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

        return model.fluxes_df["sacsma_uh_qq"]

    def evaluation(self):
        """
        Returns the observed data used for evaluation.

        Returns:
            Any: The observations associated with the current instance.
        """
        return self.observations

    def objectivefunction(self, simulation, evaluation):
        """
        Calculates the objective function value for model evaluation using
        MSE.

        This function compares observed and simulated data over selected dates,
        applies a mask to exclude NaN values, and computes the root mean 
        square error (RMSE) between the observed and simulated values. If the
        optimization algorithm is set to maximize, the RMSE value is negated.

        Parameters
        ----------
            simulation : pd.DataFrame or pd.Series)
                Simulated data indexed by date.
            evaluation : pd.DataFrame or pd.Series
                Observed data indexed by date.

        Returns
        -------
            evaluation float
                The computed objective function value (RMSE or its negative)

        To do
        ------
            - Consider adding additional objective functions for more
            comprehensive model evaluation.
        """
        obs = evaluation[self.mask_dates].values
        sim = simulation[self.mask_dates].values
        mask = ~np.isnan(obs)
        objectivefunction = rmse(obs[mask], sim[mask])
        if not self.algorithm_minimize:
            objectivefunction = -objectivefunction
        return objectivefunction
