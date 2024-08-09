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
import pystellibs
import galsim


from chromatic_weak_lensing import utils
from chromatic_weak_lensing import Stars


logger = logging.getLogger(__name__)


# Phoenix units
_wave_type = u.angstrom
_flux_type = u.erg / u.angstrom / u.second / u.pc**2
FLUX_FACTOR = (1 * _flux_type).to(galsim.SED._flambda).value


def _apply_dust(flux, wl, av):
    dust_extinction = extinction.odonnell94(wl, av, r_v=3.1, unit="aa")
    return extinction.apply(dust_extinction, flux)


def _get_spectrum(
    speclib,
    logte,
    logg,
    logl,
    metallicity,
    mu0,
    av=None,
    apply_dust=False,
):
    _start_time = time.time()

    # speclib = getattr(pystellibs, library)
    # speclib = pystellibs.BTSettl(medres=False)

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
    # if apply_dust:
    #     observed_spec = _apply_dust(observed_spec, lam, "av")
    sed_table = galsim.LookupTable(
        wl,
        spec * FLUX_FACTOR / surface_area,
    )

    sed = galsim.SED(sed_table, wave_type, flux_type)

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.debug(f"made stellar spectrum in {_elapsed_time} seconds")

    return sed


# def model_spectrum(mass):
#     ZSUN = 0.0122  # 0.0134
#     logte = np.log10(5780 * np.power(mass, 2.1 / 4))
#     logg = np.log10(1/4.13E10 * mact / _luminosity) + 4 * np.log10(_temperature)  # FSPS
#     logl = np.log10(np.power(mass, 3.5))
#     metallicity = ZSUN
#     return get_spectrum(logte, logg, logl, metallicity)


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
        av=None,
        apply_dust=False,
    ):
        return _get_spectrum(
            self.speclib,
            logte,
            logg,
            logl,
            metallicity,
            mu0,
            av=av,
            apply_dust=apply_dust,
        )
