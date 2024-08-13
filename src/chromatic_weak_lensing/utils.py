import logging
import math

import astropy.units as u
import galsim
import numpy as np
import pyarrow as pa
from dust_extinction.parameter_averages import G23


logger = logging.getLogger(__name__)


def unwrap(value):
    if isinstance(value, pa.Scalar):
       return value.as_py()
    else:
       return value


def get_distance(distance_modulus):
    return 10**(1 + distance_modulus / 5)


def get_surface_area(distance_modulus):
    """
    Area of the sphere whose radius extends from the source to the observer
    """
    # distance = 10**(1 + distance_modulus / 5)
    distance = get_distance(distance_modulus)
    return 4 * math.pi * distance**2


def get_extinction(av, red_limit=12_000):
    ext = G23(Rv=3.1)

    eps = 1e-7  # machine epsilon
    wl_min = 1e4 / ext.x_range[1]  + eps # 1/micron to angstrom
    wl_max = 1e4 / ext.x_range[0]  - eps # 1/micron to angstrom
    if wl_max > red_limit:
        wl_max = red_limit

    n_wl = int(wl_max - wl_min) // 10
    wls = np.linspace(wl_min, wl_max, n_wl)

    extinction = ext.extinguish(wls * u.AA, Av=av)

    extinction_table = galsim.LookupTable(
        wls,
        extinction,
    )
    extinction_sed = galsim.SED(
        extinction_table,
        wave_type="angstrom",
        flux_type="1",
    )

    return extinction_sed
