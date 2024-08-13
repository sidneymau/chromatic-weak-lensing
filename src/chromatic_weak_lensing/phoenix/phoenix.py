"""
Adapted from Claire-Alice (@cahebert)
"""

import functools
import logging
import math
import os
from pathlib import Path
import time

import astropy.units as u
import galsim
import pystellibs


from chromatic_weak_lensing import utils
from chromatic_weak_lensing import Stars


logger = logging.getLogger(__name__)


# Phoenix units
_wave_type = u.angstrom
_flux_type = u.erg / u.nm / u.second / u.pc**2
FLUX_FACTOR = (1 * _flux_type).to(galsim.SED._flambda).value


def _get_spectrum(
    speclib,
    logte,
    logg,
    logl,
    metallicity,
    mu0,
):
    _start_time = time.time()

    wave_type = "angstrom"
    flux_type = "flambda"

    wl = speclib._wavelength
    spec = speclib.generate_stellar_spectrum(
        logte,
        logg,
        logl,
        metallicity,
    )
    surface_area =  utils.get_surface_area(mu0)
    sed_table = galsim.LookupTable(
        wl,
        spec * FLUX_FACTOR / surface_area,
    )

    sed = galsim.SED(sed_table, wave_type, flux_type)

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.debug(f"made stellar spectrum in {_elapsed_time} seconds")

    return sed


class Phoenix(Stars):
    def __init__(self):
        self.name = "Phoenix"
        self.speclib = pystellibs.Phoenix()

    def get_spectrum(
        self,
        logte,
        logg,
        logl,
        metallicity,
        mu0,
    ):
        return _get_spectrum(
            self.speclib,
            logte,
            logg,
            logl,
            metallicity,
            mu0,
        )
