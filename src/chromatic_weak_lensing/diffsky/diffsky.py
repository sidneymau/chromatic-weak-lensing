import logging
import math
import os
import time

import astropy.units as u
import galsim
import numpy as np

import dsps
from lsstdesc_diffsky.defaults import OUTER_RIM_COSMO_PARAMS
from lsstdesc_diffsky.sed.disk_bulge_sed_kernels_singlemet import calc_rest_sed_disk_bulge_knot_singlegal


from chromatic_weak_lensing import utils
from chromatic_weak_lensing import Galaxies
from .utils import load_dsps_ssp_data, load_diffskypop_params


logger = logging.getLogger(__name__)


# DSPS units
_wave_type = u.angstrom
_flux_type = u.Lsun / u.Hz / u.Mpc**2
FLUX_FACTOR = (1 * _flux_type).to(galsim.SED._fnu).value  # 4.0204145742268754e-16


def _get_morphology(
    spheroidEllipticity1,
    spheroidEllipticity2,
    spheroidHalfLightRadiusArcsec,
    diskEllipticity1,
    diskEllipticity2,
    diskHalfLightRadiusArcsec,
    n_knots=0,
):
    _start_time = time.time()
    bulge_ellipticity = galsim.Shear(
        g1=spheroidEllipticity1,
        g2=spheroidEllipticity2,
    )
    bulge = galsim.DeVaucouleurs(
        half_light_radius=spheroidHalfLightRadiusArcsec,
    ).shear(bulge_ellipticity)

    disk_ellipticity = galsim.Shear(
        g1=diskEllipticity1,
        g2=diskEllipticity2,
    )
    disk = galsim.Exponential(
        half_light_radius=diskHalfLightRadiusArcsec,
    ).shear(disk_ellipticity)

    if n_knots > 0:
        knots = galsim.RandomKnots(
            n_knots,
            profile=disk,
        )
    else:
        knots = None

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.debug(f"made galaxy morphology in {_elapsed_time} seconds")

    return bulge, disk, knots


def _get_total_morphology(
    spheroidEllipticity1,
    spheroidEllipticity2,
    spheroidHalfLightRadiusArcsec,
    diskEllipticity1,
    diskEllipticity2,
    diskHalfLightRadiusArcsec,
    m_total,
    m_bulge,
    m_disk,
    m_knot,
    n_knots=0,
):
    bulge, disk, knots = _get_morphology(
        spheroidEllipticity1,
        spheroidEllipticity2,
        spheroidHalfLightRadiusArcsec,
        diskEllipticity1,
        diskEllipticity2,
        diskHalfLightRadiusArcsec,
        n_knots=n_knots,
    )

    if n_knots > 0:
        morphology =  (bulge * m_bulge + disk * m_disk + knots * m_knot) / m_total
    else:
        morphology = (bulge * m_bulge + disk * (m_disk + m_knot)) / m_total

    return morphology


def _get_spectrum(
     redshift,
     ssp_data,
     rest_sed_bulge,
     rest_sed_diffuse_disk,
     rest_sed_knot,
     cosmo_params,
):
    _start_time = time.time()

    luminosity_distance = dsps.cosmology.luminosity_distance_to_z(
        redshift,
        cosmo_params.Om0,
        cosmo_params.w0,
        cosmo_params.wa,
        cosmo_params.h,
    )
    surface_area = 4 * math.pi * luminosity_distance**2

    wave_type = "angstrom"
    flux_type = "fnu"

    bulge_sed_table = galsim.LookupTable(
        ssp_data.ssp_wave,
        rest_sed_bulge / surface_area * FLUX_FACTOR,
    )
    bulge_sed = galsim.SED(
        bulge_sed_table,
        wave_type=wave_type,
        flux_type=flux_type,
        redshift=redshift,
    )

    disk_sed_table = galsim.LookupTable(
        ssp_data.ssp_wave,
        rest_sed_diffuse_disk / surface_area * FLUX_FACTOR,
    )
    disk_sed = galsim.SED(
        disk_sed_table,
        wave_type=wave_type,
        flux_type=flux_type,
        redshift=redshift,
    )

    knot_sed_table = galsim.LookupTable(
        ssp_data.ssp_wave,
        rest_sed_knot / surface_area * FLUX_FACTOR,
    )
    knot_sed = galsim.SED(
        knot_sed_table,
        wave_type=wave_type,
        flux_type=flux_type,
        redshift=redshift,
    )

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.debug(f"made galaxy spectrum in {_elapsed_time} seconds")

    return bulge_sed, disk_sed, knot_sed


def _get_total_spectrum(
    redshift,
    ssp_data,
    rest_sed_bulge,
    rest_sed_diffuse_disk,
    rest_sed_knot,
    cosmo_params,
):
    bulge_sed, disk_sed, knot_sed = _get_spectrum(
        redshift,
        ssp_data,
        rest_sed_bulge,
        rest_sed_diffuse_disk,
        rest_sed_knot,
        cosmo_params,
    )
    return bulge_sed + disk_sed + knot_sed


def _get_galaxy(
    redshift,
    spheroidEllipticity1,
    spheroidEllipticity2,
    spheroidHalfLightRadiusArcsec,
    diskEllipticity1,
    diskEllipticity2,
    diskHalfLightRadiusArcsec,
    ssp_data,
    rest_sed_bulge,
    rest_sed_diffuse_disk,
    rest_sed_knot,
    cosmo_params,
    n_knots=0,
    morphology="chromatic",
    m_total=None,
    m_bulge=None,
    m_disk=None,
    m_knot=None,
):
    """
    Create a galaxy from a diffsky catalog
    """
    _start_time = time.time()

    bulge, disk, knots = _get_morphology(
        spheroidEllipticity1,
        spheroidEllipticity2,
        spheroidHalfLightRadiusArcsec,
        diskEllipticity1,
        diskEllipticity2,
        diskHalfLightRadiusArcsec,
        n_knots=n_knots,
    )

    # get restframe SEDs and redshift composite object at end
    bulge_sed, disk_sed, knot_sed = _get_spectrum(
        redshift,
        ssp_data,
        rest_sed_bulge,
        rest_sed_diffuse_disk,
        rest_sed_knot,
        cosmo_params,
    )

    match morphology:
        case "chromatic":
            # each morphological component has proper spectrum
            if n_knots > 0:
                gal = bulge * bulge_sed + disk * disk_sed + knots * knot_sed
            else:
                # if no knots are drawn, reweight the disk
                # to preserve relative magnitudes
                gal = bulge * bulge_sed + disk * (disk_sed + knot_sed)
        case "achromatic":
            # decouple the morphology from the spectra
            if n_knots > 0:
                gal = (
                    (bulge * m_bulge + disk * m_disk + knots * m_knot) / m_total \
                    * (bulge_sed + disk_sed + knot_sed)
                )
            else:
                gal = (
                    (bulge * m_bulge + disk * (m_disk + m_knot)) / m_total \
                    * (bulge_sed + disk_sed + knot_sed)
                )
        case _:
            raise ValueError("Unrecognized morphology: %s" % morphology)

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time

    logger.debug(f"made galaxy in {_elapsed_time} seconds")

    return gal


class Diffsky(Galaxies):
    cosmo_params = OUTER_RIM_COSMO_PARAMS

    def __init__(self, red_limit=12_000):
        self.name = "diffsky"
        self.diffskypop_params = load_diffskypop_params()

        _ssp_data = load_dsps_ssp_data()
        if red_limit is not None:
            from lsstdesc_diffsky.legacy.roman_rubin_2023.dsps.data_loaders.defaults import SSPDataSingleMet
            # note that we can't impose a lower limit due to redshifting
            _keep = (_ssp_data.ssp_wave < red_limit)
            logger.info(f"discarding {(~_keep).sum()} of {len(_keep)} wavelengths from templates")
            _ssp_wave = _ssp_data.ssp_wave[_keep]
            _ssp_flux = _ssp_data.ssp_flux[:, _keep]
            ssp_data = SSPDataSingleMet(_ssp_data.ssp_lg_age_gyr, _ssp_wave, _ssp_flux)
        else:
            ssp_data = _ssp_data
        self.ssp_data = ssp_data

    def get_color(
        self,
        obs_params,
    ):
        return obs_params.LSST_obs_g - obs_params.LSST_obs_i

    def get_morphology(
        self,
        morphology_params,
    ):
        disk_bulge_sed_info = calc_rest_sed_disk_bulge_knot_singlegal(
            morphology_params.redshift,
            morphology_params.diffsky_param_data.mah_params,
            morphology_params.diffsky_param_data.ms_params,
            morphology_params.diffsky_param_data.q_params,
            morphology_params.diffsky_param_data.fbulge_params,
            morphology_params.diffsky_param_data.fknot,
            self.ssp_data,
            self.diffskypop_params,
            self.cosmo_params,
        )

        return _get_total_morphology(
            morphology_params.spheroidEllipticity1,
            morphology_params.spheroidEllipticity2,
            morphology_params.spheroidHalfLightRadiusArcsec,
            morphology_params.diskEllipticity1,
            morphology_params.diskEllipticity2,
            morphology_params.diskHalfLightRadiusArcsec,
            disk_bulge_sed_info.mstar_total,
            disk_bulge_sed_info.mstar_bulge,
            disk_bulge_sed_info.mstar_diffuse_disk,
            disk_bulge_sed_info.mstar_knot,
            n_knots=morphology_params.n_knots,
        )

    def get_spectrum(
        self,
        spectrum_params,
    ):
        disk_bulge_sed_info = calc_rest_sed_disk_bulge_knot_singlegal(
            spectrum_params.redshift,
            spectrum_params.diffsky_param_data.mah_params,
            spectrum_params.diffsky_param_data.ms_params,
            spectrum_params.diffsky_param_data.q_params,
            spectrum_params.diffsky_param_data.fbulge_params,
            spectrum_params.diffsky_param_data.fknot,
            self.ssp_data,
            self.diffskypop_params,
            self.cosmo_params,
        )

        rest_sed_bulge = disk_bulge_sed_info.rest_sed_bulge
        rest_sed_diffuse_disk = disk_bulge_sed_info.rest_sed_diffuse_disk
        rest_sed_knot = disk_bulge_sed_info.rest_sed_knot

        return _get_total_spectrum(
            spectrum_params.redshift,
            self.ssp_data,
            rest_sed_bulge,
            rest_sed_diffuse_disk,
            rest_sed_knot,
            self.cosmo_params,
        )

    def get_galaxy(
        self,
        galaxy_params,
        morphology="chromatic",
    ):

        disk_bulge_sed_info = calc_rest_sed_disk_bulge_knot_singlegal(
            galaxy_params.redshift,
            galaxy_params.diffsky_param_data.mah_params,
            galaxy_params.diffsky_param_data.ms_params,
            galaxy_params.diffsky_param_data.q_params,
            galaxy_params.diffsky_param_data.fbulge_params,
            galaxy_params.diffsky_param_data.fknot,
            self.ssp_data,
            self.diffskypop_params,
            self.cosmo_params,
        )

        rest_sed_bulge = disk_bulge_sed_info.rest_sed_bulge
        rest_sed_diffuse_disk = disk_bulge_sed_info.rest_sed_diffuse_disk
        rest_sed_knot = disk_bulge_sed_info.rest_sed_knot

        mstar_total = disk_bulge_sed_info.mstar_total
        mstar_bulge = disk_bulge_sed_info.mstar_bulge
        mstar_diffuse_disk = disk_bulge_sed_info.mstar_diffuse_disk
        mstar_knot = disk_bulge_sed_info.mstar_knot

        return _get_galaxy(
            galaxy_params.redshift,
            galaxy_params.spheroidEllipticity1,
            galaxy_params.spheroidEllipticity2,
            galaxy_params.spheroidHalfLightRadiusArcsec,
            galaxy_params.diskEllipticity1,
            galaxy_params.diskEllipticity2,
            galaxy_params.diskHalfLightRadiusArcsec,
            self.ssp_data,
            rest_sed_bulge,
            rest_sed_diffuse_disk,
            rest_sed_knot,
            self.cosmo_params,
            n_knots=galaxy_params.n_knots,
            morphology=morphology,
            m_total=mstar_total,
            m_bulge=mstar_bulge,
            m_disk=mstar_diffuse_disk,
            m_knot=mstar_knot,
        )
