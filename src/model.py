import pandas as pd
import numpy as np
import src.sacsma_source.sac.ex_sac1 as sacsma
import src.sacsma_source.sac.duamel as unit_hydrograph
import src.sacsma_source.snow19.exsnow as snow17
from dataclasses import dataclass
from src.et import get_priestley_taylor_pet, get_atmospheric_pressure


@dataclass
class Sacsma_parameters:
    uztwm: float = None
    uzfwm: float = None
    uzk: float = None
    pctim: float = None
    adimp: float = None
    riva: float = None
    zperc: float = None
    rexp: float = None
    lztwm: float = None
    lzfsm: float = None
    lzfpm: float = None
    lzsk: float = None
    lzpk: float = None
    pfree: float = None
    side: float = None
    rserv: float = None


@dataclass
class Snow17_parameters:
    scf: float = None
    mfmax: float = None
    mfmin: float = None
    uadj: float = None
    si: float = None
    nmf: float = None
    tipm: float = None
    mbase: float = None
    pxtemp: float = None
    plwhc: float = None
    daygm: float = None


@dataclass
class Hydrograph_parameters:
    unit_shape: float = None
    unit_scale: float = None


@dataclass
class Sacsma_states:
    sacsma_snow17_tprev: np.ndarray = None
    sacsma_uztwc: np.ndarray = None
    sacsma_uzfwc: np.ndarray = None
    sacsma_lztwc: np.ndarray = None
    sacsma_lzfsc: np.ndarray = None
    sacsma_lzfpc: np.ndarray = None
    sacsma_adimc: np.ndarray = None


@dataclass
class Sacma_initial_states:
    cs: np.ndarray = None
    tprev: float = None
    uztwc: float = None
    uzfwc: float = None
    lztwc: float = None
    lzfsc: float = None
    lzfpc: float = None
    adimc: float = None


@dataclass
class Sacsma_fluxes:
    sacsma_pet: np.ndarray = None
    sacsma_snow17_raim: np.ndarray = None
    sacsma_snow17_sneqv: np.ndarray = None
    sacsma_snow17_snow: np.ndarray = None
    sacsma_snow17_snowh: np.ndarray = None
    sacsma_surf: np.ndarray = None
    sacsma_grnd: np.ndarray = None
    sacsma_qq: np.ndarray = None
    sacsma_tet: np.ndarray = None
    hydrograph_qq: np.ndarray = None  # This will be set later in the run method


@dataclass
class Acd_raw:
    adc1: float = None
    adc2: float = None
    adc3: float = None
    adc4: float = None
    adc5: float = None
    adc6: float = None
    adc7: float = None
    adc8: float = None
    adc9: float = None
    adc10: float = None
    adc11: float = None


@dataclass
class Acd_adjusted:
    adc1: float = 0.05
    adc2: float = 0.15
    adc3: float = 0.26
    adc4: float = 0.45
    adc5: float = 0.5
    adc6: float = 0.56
    adc7: float = 0.61
    adc8: float = 0.65
    adc9: float = 0.69
    adc10: float = 0.82
    adc11: float = 1.0


@dataclass
class Forcings:
    date: np.ndarray = None
    year: np.ndarray = None
    month: np.ndarray = None
    day: np.ndarray = None
    precip: np.ndarray = None
    max_temp: np.ndarray = None
    min_temp: np.ndarray = None
    avg_temp: np.ndarray = None
    srad: np.ndarray = None
    dayl: np.ndarray = None
    pet: np.ndarray = None
    surf_pres: float = None


@dataclass
class General_parameters:
    latitude = None
    elevation = None
    alpha_pt: float = 1.26


@dataclass
class Others:
    dt_seconds: int = None
    dt_minutes: float = None
    dt_hours: float = None
    dt_days: float = None
    m_unit_hydro: int = 1000
    n_unit_hydro: int = None  # to be set later based on forcings
    k: int = 1
    ntau: int = 0


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

adc_keys = [  # Aereal depletion curve parameters
    "adc1",
    "adc2",
    "adc3",
    "adc4",
    "adc5",
    "adc6",
    "adc7",
    "adc8",
    "adc9",
    "adc10",
    "adc11",
]


class Sacsma:

    def __init__(self):
        self.forcings = Forcings()
        self.parameters = Sacsma_parameters()
        self.snow17_parameters = Snow17_parameters()
        self.hydrograph_parameters = Hydrograph_parameters()
        self.states = Sacsma_states()
        self.initial_states = Sacma_initial_states()
        self.fluxes = Sacsma_fluxes()
        self.adc_raw = Acd_raw()
        self.adc_adjusted = Acd_adjusted()
        self.general_parameters = General_parameters()
        self.others = Others()
        self.fluxes_df = "Run model first"
        self.states_df = "Run model first"
        self.last_cs = np.full(19, 0.0, dtype="f4")

    def set_parameters(self, parameters: Sacsma_parameters):
        for field in sacsma_parameter_keys:
            value = float(getattr(parameters, field))
            if value is not None:
                setattr(self.parameters, field, value)

    def set_snow17_parameters(self, parameters: Snow17_parameters):
        for field in snow17_parameter_keys:
            value = float(getattr(parameters, field))
            if value is not None:
                setattr(self.snow17_parameters, field, value)

    def set_hydrograph_parameters(self, parameters: Hydrograph_parameters):
        for field in hydrograph_parameter_keys:
            value = float(getattr(parameters, field))
            if value is not None:
                setattr(self.hydrograph_parameters, field, value)

    def set_adc_parameters(self, parameters: Acd_raw):
        for field in adc_keys:
            value = float(getattr(parameters, field))
            if value is not None:
                setattr(self.adc_raw, field, value)

        total_adc = sum([getattr(self.adc_raw, f"adc{i}") for i in range(1, 12)])
        cum_adc = np.cumsum([getattr(self.adc_raw, f"adc{i}") for i in range(1, 12)])
        for i in range(1, 12):
            setattr(self.adc_adjusted, f"adc{i}", cum_adc[i - 1] / total_adc)

    def set_forcings(self, date, precip, max_temp, min_temp, srad, dayl):
        if not pd.api.types.is_datetime64_any_dtype(date):
            raise TypeError("Date must be a pandas datetime Series")

        self.forcings.date = date
        self.forcings.year = date.dt.year.to_numpy()
        self.forcings.month = date.dt.month.to_numpy()
        self.forcings.day = date.dt.day.to_numpy()
        self.forcings.precip = precip
        self.forcings.max_temp = max_temp
        self.forcings.min_temp = min_temp
        self.forcings.avg_temp = 0.5 * (self.forcings.max_temp + self.forcings.min_temp)
        self.forcings.srad = srad
        self.forcings.dayl = dayl

        # timestep in different units
        self.others.dt_seconds = int(
            (self.forcings.date[1] - self.forcings.date[0]).total_seconds()
        )
        self.others.dt_minutes = self.others.dt_seconds / 60
        self.others.dt_hours = self.others.dt_minutes / 60
        self.others.dt_days = self.others.dt_hours / 24

        # set an empty np.ndarray for states and fluxes
        for field in state_keys:
            setattr(self.states, field, np.zeros_like(self.forcings.precip))

        for field in flux_keys:
            setattr(self.fluxes, field, np.zeros_like(self.forcings.precip))

    def set_initial_states(self, cs, tprev, uztwc, uzfwc, lztwc, lzfsc, lzfpc, adimc):
        self.initial_states.cs = cs
        self.initial_states.tprev = tprev
        self.initial_states.uztwc = uztwc
        self.initial_states.uzfwc = uzfwc
        self.initial_states.lztwc = lztwc
        self.initial_states.lzfsc = lzfsc
        self.initial_states.lzfpc = lzfpc
        self.initial_states.adimc = adimc

    def compute_et(self, alpha_pt=None):
        if alpha_pt is None:
            alpha_pt = self.general_parameters.alpha_pt
        if (
            self.general_parameters.latitude is None
            or self.general_parameters.elevation is None
        ):
            raise ValueError("Latitude and elevation must be set in general parameters")

        self.forcings.pet = self.forcings.pet = get_priestley_taylor_pet(
            t_max=self.forcings.max_temp,
            t_min=self.forcings.min_temp,
            r_s=self.forcings.srad,
            dayl=self.forcings.dayl,
            j=self.forcings.date.dt.dayofyear,
            lat=self.general_parameters.latitude,
            elev=self.general_parameters.elevation,
            alpha_pt=alpha_pt,
        )

        self.forcings.surf_pres = get_atmospheric_pressure(
            self.general_parameters.elevation
        )

    def run(self):

        # set states from initial states
        cs = np.array(self.initial_states.cs, dtype="f4")
        tprev = np.array(self.initial_states.tprev, dtype="f4")
        uztwc = np.array(self.initial_states.uztwc, dtype="f4")
        uzfwc = np.array(self.initial_states.uzfwc, dtype="f4")
        lztwc = np.array(self.initial_states.lztwc, dtype="f4")
        lzfsc = np.array(self.initial_states.lzfsc, dtype="f4")
        lzfpc = np.array(self.initial_states.lzfpc, dtype="f4")
        adimc = np.array(self.initial_states.adimc, dtype="f4")

        for t in range(self.forcings.date.shape[0]):
            # snow17
            raim, sneqv, snow, snowh = snow17.exsnow19(
                self.others.dt_seconds,
                self.others.dt_hours,
                self.forcings.day[t],
                self.forcings.month[t],
                self.forcings.year[t],
                self.forcings.precip[t],
                self.forcings.avg_temp[t],
                self.general_parameters.latitude,
                self.snow17_parameters.scf,
                self.snow17_parameters.mfmax,
                self.snow17_parameters.mfmin,
                self.snow17_parameters.uadj,
                self.snow17_parameters.si,
                self.snow17_parameters.nmf,
                self.snow17_parameters.tipm,
                self.snow17_parameters.mbase,
                self.snow17_parameters.pxtemp,
                self.snow17_parameters.plwhc,
                self.snow17_parameters.daygm,
                self.general_parameters.elevation,
                self.forcings.surf_pres,
                np.array(
                    [
                        self.adc_adjusted.adc1,
                        self.adc_adjusted.adc2,
                        self.adc_adjusted.adc3,
                        self.adc_adjusted.adc4,
                        self.adc_adjusted.adc5,
                        self.adc_adjusted.adc6,
                        self.adc_adjusted.adc7,
                        self.adc_adjusted.adc8,
                        self.adc_adjusted.adc9,
                        self.adc_adjusted.adc10,
                        self.adc_adjusted.adc11,
                    ]
                ).astype("f4"),
                cs,
                tprev,
            )

            # sacsma
            surf, grnd, qq, tet = sacsma.exsac(
                self.others.dt_seconds,
                raim,
                self.forcings.avg_temp[t],
                self.forcings.pet[t],
                self.parameters.uztwm,
                self.parameters.uzfwm,
                self.parameters.uzk,
                self.parameters.pctim,
                self.parameters.adimp,
                self.parameters.riva,
                self.parameters.zperc,
                self.parameters.rexp,
                self.parameters.lztwm,
                self.parameters.lzfsm,
                self.parameters.lzfpm,
                self.parameters.lzsk,
                self.parameters.lzpk,
                self.parameters.pfree,
                self.parameters.side,
                self.parameters.rserv,
                uztwc,
                uzfwc,
                lztwc,
                lzfsc,
                lzfpc,
                adimc,
            )

            # Saving states for this timestep
            self.states.sacsma_snow17_tprev[t] = tprev
            self.states.sacsma_uztwc[t] = uztwc
            self.states.sacsma_uzfwc[t] = uzfwc
            self.states.sacsma_lztwc[t] = lztwc
            self.states.sacsma_lzfsc[t] = lzfsc
            self.states.sacsma_lzfpc[t] = lzfpc
            self.states.sacsma_adimc[t] = adimc

            # print every 5 time steps
            # if t % 5 == 0:
            #    print(f'Timestep {t}: uztwc={uztwc:.2f}, uzfwc={uzfwc:.2f}, lztwc={lztwc:.2f}, lzfsc={lzfsc:.2f}, lzfpc={lzfpc:.2f}, adimc={adimc:.2f}')

            # Saving fluxes for this timestep
            self.fluxes.sacsma_pet[t] = self.forcings.pet[t]
            self.fluxes.sacsma_snow17_raim[t] = raim
            self.fluxes.sacsma_snow17_sneqv[t] = sneqv
            self.fluxes.sacsma_snow17_snow[t] = snow
            self.fluxes.sacsma_snow17_snowh[t] = snowh
            self.fluxes.sacsma_surf[t] = surf
            self.fluxes.sacsma_grnd[t] = grnd
            self.fluxes.sacsma_qq[t] = qq
            self.fluxes.sacsma_tet[t] = tet

        self.others.n_unit_hydro = (
            self.forcings.date.shape[0] + self.others.m_unit_hydro
        )

        hydrograph_qq = unit_hydrograph.duamel(
            self.fluxes.sacsma_qq,
            self.hydrograph_parameters.unit_shape,
            self.hydrograph_parameters.unit_scale,
            self.others.dt_days,
            self.others.n_unit_hydro,
            self.others.m_unit_hydro,
            self.others.k,
            self.others.ntau,
        )

        self.fluxes.hydrograph_qq = hydrograph_qq[: -self.others.m_unit_hydro]

        self.fluxes_df = pd.DataFrame(
            {
                "sacsma_pet": self.fluxes.sacsma_pet,
                "sacsma_snow17_raim": self.fluxes.sacsma_snow17_raim,
                "sacsma_uh_qq": self.fluxes.hydrograph_qq,
                "sacsma_snow17_sneqv": self.fluxes.sacsma_snow17_sneqv,
                "sacsma_snow17_snow": self.fluxes.sacsma_snow17_snow,
                "sacsma_snow17_snowh": self.fluxes.sacsma_snow17_snowh,
                "sacsma_surf": self.fluxes.sacsma_surf,
                "sacsma_grnd": self.fluxes.sacsma_grnd,
                "sacsma_qq": self.fluxes.sacsma_qq,
                "sacsma_tet": self.fluxes.sacsma_tet,
            },
            index=self.forcings.date,
        )

        self.states_df = pd.DataFrame(
            {
                "sacsma_snow17_tprev": self.states.sacsma_snow17_tprev,
                "sacsma_uztwc": self.states.sacsma_uztwc,
                "sacsma_uzfwc": self.states.sacsma_uzfwc,
                "sacsma_lztwc": self.states.sacsma_lztwc,
                "sacsma_lzfsc": self.states.sacsma_lzfsc,
                "sacsma_lzfpc": self.states.sacsma_lzfpc,
                "sacsma_adimc": self.states.sacsma_adimc,
            },
            index=self.forcings.date,
        )

        self.last_cs = np.array(cs, dtype="f4")

