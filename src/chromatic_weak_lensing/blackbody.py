import functools
import logging
import time

import astropy.units as u
from astropy.constants import h, c, k_B, sigma_sb, R_sun
from astropy.modeling.models import BlackBody
import math
import galsim
import numpy as np

from chromatic_weak_lensing import utils
from chromatic_weak_lensing import Stars


logger = logging.getLogger(__name__)


_wave_type = u.angstrom
_flux_type = u.erg / u.AA / u.second / u.cm**2
FLUX_FACTOR = (1 * _flux_type).to(galsim.SED._flambda).value

SOLRAD = R_sun.to(u.pc).value


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
        * (radius * SOLRAD / distance)**2
    ).to(galsim.SED._flambda).value


def _blackbody_sed(
    temperature,
    distance_modulus,
    radius,
):
    return galsim.SED(
        functools.partial(
            _blackbody_luminosity,
            temperature=temperature,
            distance_modulus=distance_modulus,
            radius=radius,
        ),
        wave_type="nm",
        flux_type="flambda"
    )


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
        return _blackbody_sed(
            temperature,
            distance_modulus,
            radius,
        )
