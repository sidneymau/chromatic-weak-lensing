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
    mass,
    logt,
    logl,
    logg,
    distance_modulus,
    zmet=None,
    phase=1,
    composition=1,
):
    _start_time = time.time()

    wave_type = "angstrom"
    flux_type = "fnu"

    lbol = 10 ** logl

    wl = stellar_population.wavelengths
    spec = stellar_population._get_stellar_spectrum(
        mass,
        logt,
        lbol,
        logg,
        phase,
        composition,
        zmet=zmet,
        peraa=False,
    )
    surface_area =  utils.get_surface_area(distance_modulus)
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
        self.stellar_population = fsps.StellarPopulation(
            zcontinuous=0,
            add_neb_emission=True,
        )
        self.spec_library = self.stellar_population.spec_library.decode("utf-8")

    def get_params(
        self,
        stellar_params,
    ):
        if stellar_params.phase is not None:
            return (
                stellar_params.mass,
                stellar_params.logT,
                stellar_params.logL,
                stellar_params.logg,
                stellar_params.distance_modulus,
                stellar_params.z,
                stellar_params.phase,
                stellar_params.composition,
            )
        else:
            return (
                stellar_params.mass,
                stellar_params.logT,
                stellar_params.logL,
                stellar_params.logg,
                stellar_params.distance_modulus,
                stellar_params.z,
            )

    def get_spectrum(
        self,
        mass,
        logt,
        logl,
        logg,
        distance_modulus,
        z,
        phase=1,
        composition=1,
    ):
        # get index of closest metallicity from legend
        # note that fortran indexes from 1 onwards, not 0
        zmet = np.argmin(
            np.abs(
                np.subtract(
                    z,
                    self.stellar_population.zlegend,
                ),
            ),
        ) + 1
        logger.info(f"using zmet={zmet} [z={self.stellar_population.zlegend[zmet - 1]}]")
        return _get_spectrum(
            self.stellar_population,
            mass,
            logt,
            logl,
            logg,
            distance_modulus,
            zmet=zmet,
            phase=phase,
            composition=composition,
        )
