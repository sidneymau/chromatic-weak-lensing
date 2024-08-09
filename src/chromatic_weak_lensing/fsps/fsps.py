"""
"""

import functools
import logging
import os
import math
import time

import astropy.units as u
import fsps
import numpy as np
import galsim


from chromatic_weak_lensing import utils
from chromatic_weak_lensing import Stars


logger = logging.getLogger(__name__)

# FSPS units
_wave_type = u.angstrom
_flux_type = u.Lsun / u.Hz / u.pc**2
FLUX_FACTOR = (1 * _flux_type).to(galsim.SED._fnu).value


def _get_spectrum(
    stellar_population,
    mact,
    logt,
    lbol,
    logg,
    mu0,
    phase=1,
    comp=0.5,
):
    _start_time = time.time()

    wave_type = "angstrom"
    flux_type = "fnu"

    wl = stellar_population.wavelengths
    spec = stellar_population._get_stellar_spectrum(
        mact,
        logt,
        lbol,
        logg,
        phase,
        comp,
        peraa=False,
    )
    surface_area =  utils.get_surface_area(mu0)
    sed_table = galsim.LookupTable(
        wl,
        spec * FLUX_FACTOR / surface_area,
    )
    sed = galsim.SED(sed_table, wave_type=wave_type, flux_type=flux_type)

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.debug(f"made stellar spectrum in {_elapsed_time} seconds")

    return sed


class FSPS(Stars):
    def __init__(self):
        self.name = "FSPS"
        self.sp = fsps.StellarPopulation(zcontinuous=1)

    def get_spectrum(
        self,
        mact,
        logt,
        lbol,
        logg,
        mu0,
        phase=1,
        comp=0.5,
    ):
        return _get_spectrum(
            self.sp,
            mact,
            logt,
            lbol,
            logg,
            mu0,
            phase=phase,
            comp=comp,
        )
