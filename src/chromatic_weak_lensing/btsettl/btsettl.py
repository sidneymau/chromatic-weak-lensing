"""
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
    logt,
    logl,
    logg,
    distance_modulus,
    z,
):
    _start_time = time.time()

    wave_type = "angstrom"
    flux_type = "flambda"

    wl = speclib._wavelength
    spec = speclib.generate_stellar_spectrum(
        logt,
        logg,
        logl,
        z,
    )
    surface_area =  utils.get_surface_area(distance_modulus)
    sed_table = galsim.LookupTable(
        wl,
        spec * FLUX_FACTOR / surface_area,
    )

    sed = galsim.SED(sed_table, wave_type, flux_type)

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.info(f"made stellar spectrum in {_elapsed_time} seconds")

    return sed


class BTSettl(Stars):
    def __init__(self):
        self.name = "BT-Settl"
        self.speclib = pystellibs.BTSettl(medres=False)

    def get_params(
        self,
        stellar_params,
    ):
        return (
            stellar_params.logT,
            stellar_params.logL,
            stellar_params.logg,
            stellar_params.distance_modulus,
            stellar_params.z,
        )

    def get_spectrum(
        self,
        logt,
        logl,
        logg,
        distance_modulus,
        z,
    ):
        return _get_spectrum(
            self.speclib,
            logt,
            logl,
            logg,
            distance_modulus,
            z,
        )
