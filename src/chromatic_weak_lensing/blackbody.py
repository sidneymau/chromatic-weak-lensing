import functools
import logging
import time

import astropy.units as u
from astropy.constants import h, c, k_B
import math
import galsim
import numpy as np

from chromatic_weak_lensing import utils
from chromatic_weak_lensing import Stars


logger = logging.getLogger(__name__)


_wave_type = u.angstrom
_flux_type = u.erg / u.AA / u.second / u.cm**2
FLUX_FACTOR = (1 * _flux_type).to(galsim.SED._flambda).value


def _blackbody_luminosity(
    wl,
    temperature=None,
    distance_modulus=None,
    radius=None,
):
    distance = utils.get_distance(distance_modulus)
    return (
        2 * math.pi * h * c**2
        / (wl * u.nm)**5
        / (np.exp(h * c / (wl * u.nm * k_B * temperature * u.K)) - 1)
        * (radius * u.R_sun / (distance * u.pc))**2
    ).to(galsim.SED._flambda).value


def _get_spectrum(
    temperature,
    distance_modulus,
    radius,
):
    _start_time = time.time()
    sed = galsim.SED(
        functools.partial(
            _blackbody_luminosity,
            temperature=temperature,
            distance_modulus=distance_modulus,
            radius=radius,
        ),
        wave_type="nm",
        flux_type="flambda"
    )

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.info(f"made stellar spectrum in {_elapsed_time} seconds")

    return sed


class Blackbody(Stars):
    def __init__(self):
        self.name = "Blackbody"

    def get_params(
        self,
        stellar_params,
    ):
        return (
            stellar_params.T,
            stellar_params.distance_modulus,
            stellar_params.radius,
        )

    def get_spectrum(
        self,
        temperature,
        distance_modulus,
        radius,
    ):
        return _get_spectrum(
            temperature,
            distance_modulus,
            radius,
        )
