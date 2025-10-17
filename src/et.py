import numpy as np

# Constants
EMISSIVITY = 0.97  # dimensionless
ALBEDO = 0.23  # dimensionless
LAMBDA_VAL = 2.45  # MJ kg^-1
CP = 1.013e-03  # MJ kg^-1 C^-1
E = 0.622  # dimensionless (ratio molecular weight of water vapour/dry air)
PSYCHROMETRIC_CONSTANT = 0.066  # kPa K^-1
G_SC = 0.0820  # MJ m^-2 h^-1
STEFAN_BOLTZMANN = 4.903e-09  # MJ K^-4 m^-2 day^-1
ALPHA_PT = 1.26  # for well-watered surfaces
GAMMA = 0.066  # kPa K^-1
KRS = 0.17


def get_esa(t_a: np.ndarray) -> np.ndarray:
    """
    Compute the saturation vapor pressure (es) from the mean air temperature (tmean).

    Equation:
    --------

    $$e_{sa} = 0.611 \cdot \exp{ \left( \frac{17.27 T_a}{T_a + 237.3}\right)}$$

    Where:
    $e_{sa}$ is the saturation vapor pressure in kPa,
    $T_a$ is the air temperature in °C.

    Arguments:
    ---------
    t_a : np.ndarray
        Air temperature in °C.

    Return
    ------
    np.ndarray
        Saturation vapor pressure in kPa.
    """
    es = 0.6108 * np.exp(17.27 * t_a / (t_a + 237.3))  # kPa
    return es  # kPa


def get_delta(t_a: np.ndarray) -> np.ndarray:
    """
    Compute the slope of the saturation vapor pressure curve (delta) from the air temperature (t_a).

    Equation:
    --------

    $$\Delta = \frac{4098 \cdot e_{sa}}{(T_a + 237.3)^2}$$

    Where:
    $\Delta$ is the slope of the saturation vapor pressure curve in kPa °C^-1^,
    $e_{sa}$ is the saturation vapor pressure in kPa,
    $T_a$ is the air temperature in °C.

    Arguments:
    ---------
    t_a : np.ndarray
        Air temperature in °C.

    Return:
    ------
    np.ndarray
        Slope of the saturation vapor pressure curve in kPa °C^-1^.
    """
    esa = get_esa(t_a)
    delta = 4098 * esa / (t_a + 237.3) ** 2
    return delta  # kPa °C^-1^


def get_lambda(t_a: np.ndarray = None, lambdaval: float = LAMBDA_VAL) -> np.ndarray:
    """
    Compute the latent heat of vaporization (lambda) from the air temperature (t_a).

    Equation:
    --------

    $$\lambda = 2.501 - 0.002361 T_a$$

    Where:
    $\lambda$ is the latent heat of vaporization in MJ kg^-1^,
    $T_a$ is the air temperature in °C.

    Arguments:
    ---------
    t_a: np.ndarray
        Air temperature in °C.
    lambdaval: float
        Latent heat of vaporization in MJ kg^-1^. If not temperature is provided

    Return
    ------
    np.ndarray
        Latent heat of vaporization in MJ kg^-1^.
    """
    if t_a is None:
        return lambdaval  # MJ kg^-1^
    else:
        return 2.501 - 0.002361 * t_a  # MJ kg^-1^


def get_gamma(t_a: np.ndarray) -> np.ndarray:
    """
    Compute the psychrometric constant (gamma) from the air temperature (t_a).

    Equation:
    --------

    $$\gamma = 0.066 \text{ kPa K}^{-1}$$

    Where:
    $\gamma$ is the psychrometric constant in kPa °C^-1^.

    Arguments:
    ---------
    t_a : np.ndarray
        Air temperature in °C.

    Return:
    ------
    np.ndarray
        Psychrometric constant in kPa °C^-1^.
    """
    return 0.066 * np.ones_like(t_a)


def get_net_shortwave(r_s: np.ndarray, albedo: float = ALBEDO) -> np.ndarray:
    """
    Compute the net shortwave radiation (R_ns) from the incoming shortwave radiation (R_s) and albedo.

    Equation:
    -------

    $$R_{ns} = (1 - \alpha) \cdot R_s$$

    Where:
    - $R_{ns}$ is the net shortwave radiation in W m^-2^,
    - $R_s$ is the incoming shortwave radiation in W m^-2^,
    - $\alpha$ is the albedo of the surface.

    Arguments:
    ---------
    r_s: np.ndarray
        Incoming shortwave radiation in W m^-2^.
    albedo: float
        Albedo of the surface.

    Return:
    ------
    np.ndarray
        Net shortwave radiation in W m^-2^.
    """
    return r_s * (1 - albedo)


def get_solar_declination(j: float) -> float:
    """
    Compute the solar declination (delta) from the day of the year (j).

    Equation:
    --------

    $$ \delta = 0.409 \sin{ \left( \frac{2\pi}{365} \cdot j - 1.39 \right)}$$

    Where:
    $\delta$ is the solar declination in radians,
    $j$ is the day of the year (1-365).

    Arguments:
    ---------
    j: float
        Day of the year (1-365).

    Return:
    ------
    float
        Solar declination in radians.
    """
    return 0.409 * np.sin(((2 * np.pi) / 365) * j - 1.39)


def get_sunset_hour_angle(lat: float, solar_declination: float) -> float:
    """
    Compute the sunset hour angle (omega_s) from the latitude (phi) and solar declination (delta).

    Equation:
    --------

    $$\omega_s = \arccos{ \left( -\tan{\phi} \cdot \tan{\delta} \right)}$$

    Where:
    $\omega_s$ is the sunset hour angle in radians,
    $\phi$ is the latitude in radians,
    $\delta$ is the solar declination in radians

    Arguments:
    ---------
    lat: float
        Latitude in radians.
    solar_declination: float
        Solar declination in radians.

    Return:
    ------
    float
        Sunset hour angle in radians.
    """
    omega_s = -np.tan(lat) * np.tan(solar_declination)
    omega_s[omega_s < -1] = -1
    omega_s[omega_s > 1] = 1
    omega_s = np.arccos(omega_s)
    return omega_s  # radians


def get_day_length(omega_s: float) -> float:
    """
    Compute the day length (L) from the sunset hour angle (omega_s).

    Equation:
    --------

    $$L = \frac{24}{\pi}\cdot \omega_s$$

    Where:
    $L$ is the day length in hours,
    $\omega_s$ is the sunset hour angle in radians.

    Arguments:
    ---------
    omega_s: float
        Sunset hour angle in radians.

    Return:
    ------
    float
        Day length in hours.
    """
    return (24 / np.pi) * omega_s


def get_distance_sun(j: np.ndarray) -> np.ndarray:
    """
    Compute the inverse relative distance from the Earth to the Sun (dr) from the day of the year (j).

    Equation:
    -------

    $$d_r = 1.0 + 0.033 \cdot \cos{ \left( \frac{2\pi}{365} \cdot j \right)}$$

    Where:
    $d_r$ is the distance from the Earth to the Sun (AU),
    $j$ is the day of the year (1-365).

    Arguments:
    ---------
    j: np.ndarray
        Day of the year (1-365).

    Return:
    ------
    np.ndarray
        Distance from the Earth to the Sun (AU).
    """
    return 1.0 + 0.033 * np.cos(((2.0 * np.pi) / 365.0) * j)


def get_extraterrestrial_radiation(
    lat: float,
    omega_s: np.ndarray,
    d_r: np.ndarray,
    delta: np.ndarray,
    g_sc: float = G_SC,
) -> np.ndarray:
    """
    Compute the extraterrestrial radiation (R_a) from the latitude (phi), sunset hour angle (omega_s),
    distance from the Earth to the Sun (d_r), and solar declination (delta).

    Equation:
    --------

    $$R_a = \frac{24 \cdot 60}{\pi} \cdot G_{sc} \cdot d_r \cdot \left( \omega_s \cdot \sin{\phi} \cdot \sin{\delta} + \cos{\phi} \cdot \cos{\delta} \cdot \sin{\omega_s} \right)$$

    Where:
    $R_a$ is the extraterrestrial radiation in MJ m^-2^ day^-1^,
    $G_{sc}$ is the solar constant in MJ m^-2^ min^-1^,
    $\phi$ is the latitude in radians,
    $\omega_s$ is the sunset hour angle in radians,
    $d_r$ is the distance from the Earth to the Sun (AU),
    $\delta$ is the solar declination in radians.

    Arguments:
    ---------
    lat: float
        Latitude in radians.
    omega_s: np.ndarray
        Sunset hour angle in radians.
    d_r: np.ndarray
        Inverse relative distance from the Earth to the Sun (AU).
    delta: np.ndarray
        Solar declination in radians.
    g_sc: float
        Solar constant in MJ m^-2^ min^-1^.

    Return:
    ------
    np.ndarray
        Extraterrestrial radiation in MJ m^-2^ day^-1^.
    """
    return (
        (24 * 60 / np.pi)
        * g_sc
        * d_r
        * (
            omega_s * np.sin(lat) * np.sin(delta)
            + np.cos(lat) * np.cos(delta) * np.sin(omega_s)
        )
    )


def get_clear_sky_sr(ra: np.ndarray, elev: float) -> np.ndarray:
    """
    Compute the clear sky solar radiation (R_s) from the extraterrestrial radiation (R_a) and elevation (elev).

    Equation:
    --------

    $$R_{so} = R_a \cdot (0.75 + 2  \times  10^{-5} \cdot elev)$$

    Where:
    $R_{so}$ is the clear sky solar radiation in MJ m^-2^ day^-1^,
    $R_a$ is the extraterrestrial radiation in MJ m^-2^ day^-1^,
    $elev$ is the elevation in meters.

    Arguments:
    ---------
    ra: np.ndarray
        Extraterrestrial radiation in MJ m^-2^ day^-1^.
    elev: float
        Elevation in meters.

    Return:
    ------
    np.ndarray
        Clear sky solar radiation in MJ m^-2^ day^-1^.
    """
    return ra * (0.75 + 2 * 10**-5 * elev)


def get_cloud_factor(r_so: np.ndarray, r_s: np.ndarray) -> np.ndarray:
    """
    Compute the cloud factor (f_c) from the clear sky solar radiation (R_so) and the actual solar radiation (R_s).

    Equation:
    --------

    $$f_c = \frac{R_s}{R_{so}}$$

    Where:
    $f_c$ is the cloud factor (dimensionless),
    $R_s$ is the actual solar radiation in MJ m^-2^ day^-1^,
    $R_{so}$ is the clear sky solar radiation in MJ m^-2^ day^-1^.

    Arguments:
    ---------
    r_so: np.ndarray
        Clear sky solar radiation in MJ m^-2^ day^-1^.
    r_s: np.ndarray
        Actual solar radiation in MJ m^-2^ day^-1^.

    Return:
    ------
    np.ndarray
        Cloud factor (dimensionless).
    """
    r_so[r_so == 0] = 0.001  # Avoid division by zero
    fc = r_s / r_so
    fc[fc < 0.3] = 0.3
    fc[fc > 1.0] = 1.0
    return fc


def get_net_longwave(
    t_max: np.ndarray,
    t_min: np.ndarray,
    e_a: np.ndarray,
    fc: np.ndarray,
    sigma: float = STEFAN_BOLTZMANN,
) -> np.ndarray:
    """
    Compute the net longwave radiation (R_nl) from the maximum temperature (T_max), minimum temperature (T_min),
    actual vapor pressure (e_a), and cloud factor (f_c).

    Equation:
    --------

    $$R_{nl} = \sigma \cdot \frac{(T_{max,K}^4 + T_{min,K}^4)}{2} \cdot (0.34 - 0.14 \cdot \sqrt{e_a}) \cdot (1.35 \cdot f_c - 0.35)$$

    Where:
    $R_{nl}$ is the net longwave radiation in MJ m^-2^ day^-1^,
    $T_{max}$ is the maximum temperature in C, (includes unit convertion)
    $T_{min}$ is the minimum temperature in C, (includes unit convertion)
    $e_a$ is the actual vapor pressure in kPa,
    $f_c$ is the cloud factor (dimensionless),
    $\sigma$ is the Stefan-Boltzmann constant in MJ K^-4 m^-2 day^-1^.

    Arguments:
    ---------
    t_max: np.ndarray
        Maximum temperature in K.
    t_min: np.ndarray
        Minimum temperature in K.
    e_a: np.ndarray
        Actual vapor pressure in kPa.
    fc: np.ndarray
        Cloud factor (dimensionless).
    sigma: float
        Stefan-Boltzmann constant in MJ K^-4 m^-2 day^-1^.

    Return:
    ------
    np.ndarray
        Net longwave radiation in MJ m^-2^ day^-1^.
    """
    t_max_k = t_max + 273.15
    t_min_k = t_min + 273.15
    return (
        sigma
        * 0.5
        * (t_max_k**4 + t_min_k**4)
        * (0.34 - 0.14 * np.sqrt(np.maximum(e_a, 0.0)))
        * (1.35 * fc - 0.35)
    )


def get_net_radiation(r_ns: np.ndarray, r_nl: np.ndarray) -> np.ndarray:
    """
    Compute the net radiation (R_n) from the net shortwave radiation (R_ns) and the net longwave radiation (R_nl).

    Equation:
    --------

    $$R_n = R_{ns} - R_{nl}$$

    Where:
    $R_n$ is the net radiation in MJ m^-2^ day^-1^,
    $R_{ns}$ is the net shortwave radiation in MJ m^-2^ day^-1^,
    $R_{nl}$ is the net longwave radiation in MJ m^-2^ day^-1^.

    Arguments:
    ---------
    r_ns: np.ndarray
        Net shortwave radiation in MJ m^-2^ day^-1^.
    r_nl: np.ndarray
        Net longwave radiation in MJ m^-2^ day^-1^.

    Return:
    ------
    np.ndarray
        Net radiation in MJ m^-2^ day^-1^.
    """
    return r_ns - r_nl


def get_atmospheric_pressure(elev: float) -> float:
    """
    Compute the atmospheric pressure (P) from the elevation (elev).

    Equation:
    --------

    $$P = 101.3 \left( \frac{293 - 0.0065 \cdot elev}{293} \right)^{5.26}$$

    Where:
    $P$ is the atmospheric pressure in kPa,
    $elev$ is the elevation in meters above sea level.

    Arguments:
    ---------
    elev: float
        Elevation in meters above sea level.

    Return:
    ------
    float
        Atmospheric pressure in kPa.
    """
    return 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26


def get_gamma(
    atm_press: float, cp: float = CP, lambda_: float = LAMBDA_VAL, e: float = E
) -> float:
    """
    Compute the psychrometric constant (gamma) from the atmospheric pressure (atm_press).

    Equation:
    --------

    $$\gamma = \frac{c_p \cdot P}{\varepsilon \cdot \lambda}$$

    Where:
    $\gamma$ is the psychrometric constant in kPa K^-1^,
    $P$ is the atmospheric pressure in kPa,
    $c_p$ specific heat at constant pressure in MJ kg^-1 C^-1^,
    $\lambda$ is the latent heat of vaporization in MJ kg^-1^.
    $\varepsilon$ is the ratio of the molecular weight of water vapor to dry air (dimensionless).

    Arguments:
    ---------
    atm_press: float
        Atmospheric pressure in kPa.
    cp: float
        Specific heat at constant pressure in MJ kg^-1 C^-1^.
    lambda_: float
        Latent heat of vaporization in MJ kg^-1^.
    e: float
        Ratio of the molecular weight of water vapor to dry air (dimensionless).

    Return:
    ------
    float
        Psychrometric constant in kPa K^-1^.
    """
    return (cp * atm_press) / (e * lambda_)


def get_priestley_taylor_pet(
    t_max: np.ndarray,
    t_min: np.ndarray,
    r_s: np.ndarray,
    dayl: np.ndarray,
    j: np.ndarray,
    lat: float,
    elev: float,
    gamma: float = GAMMA,
    alpha_pt: float = ALPHA_PT,
    g: np.ndarray = 0,
):
    """
    Compute the Priestley-Taylor potential evapotranspiration (PET) from the maximum temperature (T_max), minimum temperature (T_min),
    solar radiation (R_s), and other climatic variables.

    Equation:
    --------

    $$PET = \frac{\alpha_{P-T}}{\lambda} \cdot \frac{\Delta (R_n - G)}{\Delta + \gamma}$$

    Where:
    $PET$ is the potential evapotranspiration in mm day^-1^,
    $\alpha_{P-T}$ is the Priestley-Taylor coefficient (dimensionless),
    $\lambda$ is the latent heat of vaporization in MJ kg^-1^,
    $\Delta$ is the slope of the saturation vapor pressure curve in kPa K^-1^,
    $R_n$ is the net radiation in MJ m^-2^ day^-1^,
    $G$ is the soil heat flux density in MJ m^-2^ day^-1^,
    $\gamma$ is the psychrometric constant in kPa K^-1^.

    Arguments:
    ---------
    t_max: np.ndarray
        Maximum temperature in °C.
    t_min: np.ndarray
        Minimum temperature in °C.
    r_s: np.ndarray
        Solar radiation in MJ m^-2^ day^-1^.
    dayl: np.ndarray
        Day length in seconds.
    j: np.ndarray
        Julian day.
    lat: float
        Latitude in decimal degrees.
    elev: float
        Elevation in meters above sea level.
    gamma: float
        Psychrometric constant in kPa K^-1^.
    alpha_pt: float
        Priestley-Taylor coefficient (dimensionless).
    g: np.ndarray
        Soil heat flux density in MJ m^-2^ day^-1^.
    """

    # latitude conversion from degrees to radians
    lat = lat * np.pi / 180

    # Slope of saturation vapour pressure curve
    delta = get_delta(t_a=(t_max + t_min) / 2)

    # incoming netto short-wave radiation
    r_s = (
        r_s * dayl / 1000000
    )  # solar radiation conversion from W m-2 to MJ m-2 day-1 (different)
    r_ns = get_net_shortwave(r_s)

    # outgoginng netto long-wave radiation
    d_s = get_solar_declination(j)
    omega_s = get_sunset_hour_angle(lat, d_s)
    d_r = get_distance_sun(j)
    r_a = get_extraterrestrial_radiation(lat, omega_s, d_r, d_s)
    r_so = get_clear_sky_sr(r_a, elev)
    e_a = get_esa(t_min)
    f_c = get_cloud_factor(r_so, r_s)  # cloud factor from .3 to 1 (different)
    r_nl = get_net_longwave(t_max, t_min, e_a, f_c)

    # net radiation
    r_n = get_net_radiation(r_ns, r_nl)

    # gamma
    atm_press = get_atmospheric_pressure(elev)
    gamma = get_gamma(atm_press)

    # lambda
    lambda_ = get_lambda(t_a=(t_max + t_min) / 2)

    pet = (alpha_pt / lambda_) * (delta * (r_n - g)) / (delta + gamma)
    pet = np.maximum(pet, 0)
    return pet


def get_hargreaves_samani_pet(
    t_min: np.ndarray, t_max: np.ndarray, lat: float, j: float
) -> np.ndarray:
    """
    Compute the Hargreaves-Samani (HS) potential evapotranspiration (PET) using the Hargreaves equation.

    Equation:
    --------

    $$PET = 0.0135 \times KT \times (T_a + 17.8)\times (T_{max} - T_{min})^{0.5} \times Ra

    Where:
    ------
    $PET$: Potential evapotranspiration in mm day^-1^.
    $K_T$: Temperature correction factor (dimensionless).
    $T_a$: Mean air temperature in °C.
    $R_a$: Extraterrestrial radiation in MJ m^-2 day^-1.
    $T_{max}$: Maximum temperature in °C.
    $T_{min}$: Minimum temperature in °C.

    Arguments:
    ---------
    t_min: np.ndarray
        Minimum temperature in °C.
    t_max: np.ndarray
        Maximum temperature in °C.
    lat: float
        Latitude in decimal degrees.
    j: float
        Julian day.

    """

    t_a = (t_max + t_min) / 2
    lambda_ = get_lambda(t_a=t_a)

    d_s = get_solar_declination(j)
    omega_s = get_sunset_hour_angle(lat, d_s)
    d_r = get_distance_sun(j)
    r_a = get_extraterrestrial_radiation(lat, omega_s, d_r, d_s)

    pet = 0.0135 * KRS * r_a / lambda_ * (t_max - t_min) ** 0.5 * (t_a + 17.8)
    pet = np.maximum(pet, 0)
    return pet
