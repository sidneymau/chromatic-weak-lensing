import functools
import logging
import time

import astropy.units as u
from astropy.constants import h, c, k_B, sigma_sb
import math
import galsim
import numpy as np


from chromatic_weak_lensing import utils
from chromatic_weak_lensing import Stars


logger = logging.getLogger(__name__)


def _blackbody_radiance(t, mu0, wl):
    """
    See https://en.wikipedia.org/wiki/Planck%27s_law#Different_forms
    Returns blackbody radiance in flambda [u.erg / u.nm / u.cm**2 / u.s]
    """
    surface_area =  utils.get_surface_area(mu0)
    return (
        2 * h * c**2
        / (wl * u.nm)**5
        / (np.exp(h * c / (wl * u.nm * k_B * t * u.K)) - 1)
    ).to(galsim.SED._flambda).value / surface_area

    # # Flambda ?
    # mu0 = 18
    # r = 10**(1 + mu0 / 5)
    # R = u.Rsun.to(u.pc)
    # return (
    #     2 * math.pi * h * c**2
    #     / (wl * u.nm)**5
    #     / (np.exp(h * c / (wl * u.nm * k_B * t * u.K)) - 1)
    #     * (R / r)**2
    # ).to(galsim.SED._flambda).value


def _blackbody_sed(temp, mu0):
    return galsim.SED(
        functools.partial(_blackbody_radiance, temp, mu0),
        wave_type="nm",
        flux_type="flambda"
    )


class Blackbody(Stars):
    def __init__(self):
        self.name = "Blackbody"

    def get_spectrum(self, temp, mu0):
        return _blackbody_sed(temp, mu0)
