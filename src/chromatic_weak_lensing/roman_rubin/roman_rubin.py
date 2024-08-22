import logging
import math
import os
import time

import numpy as np

from lsstdesc_diffsky.io_utils.load_diffsky_healpixel import ALL_DIFFSKY_PNAMES
from lsstdesc_diffsky.constants import (
    BURSTSHAPE_PNAMES,
    FBULGE_PNAMES,
    MAH_PNAMES,
    MS_PNAMES,
    Q_PNAMES,
)
from lsstdesc_diffsky.io_utils.load_diffsky_healpixel import DiffskyParams


from chromatic_weak_lensing import utils


logger = logging.getLogger(__name__)


# from https://github.com/LSSTDESC/skyCatalogs/blob/main/skycatalogs/objects/diffsky_object.py
def _get_knot_n(data, i, seed=None):
    """
    Return random value for number of knots based on galaxy sm.
    """
    rng = np.random.default_rng(seed)
    ud = rng.uniform()

    um_source_galaxy_obs_sm = utils.unwrap(data["um_source_galaxy_obs_sm"][i])

    sm = np.log10(um_source_galaxy_obs_sm)
    m = (50 - 3) / (12 - 6)  # (knot_n range)/(logsm range)
    n_knot_max = m * (sm - 6) + 3
    n_knot = int(ud * n_knot_max)  # random n up to n_knot_max

    logger.info(f"simulating {n_knot} knots for galaxy {i}")

    return n_knot


def _get_diffsky_params(data, i):
    mah_params = np.array([utils.unwrap(data[key][i]) for key in MAH_PNAMES])
    ms_params = np.array([utils.unwrap(data[key][i]) for key in MS_PNAMES])
    q_params = np.array([utils.unwrap(data[key][i]) for key in Q_PNAMES])
    fburst = np.array(utils.unwrap(data["fburst"][i]))
    burstshape_params = np.array([utils.unwrap(data[key][i]) for key in BURSTSHAPE_PNAMES])
    fbulge_params = np.array([utils.unwrap(data[key][i]) for key in FBULGE_PNAMES])
    fknot = np.array(utils.unwrap(data["fknot"][i]))

    diffsky_param_data = DiffskyParams(
        mah_params, ms_params, q_params, fburst, burstshape_params, fbulge_params, fknot
    )

    return diffsky_param_data


class RomanRubin:
    # these are the minimal necessary columns for producing diffsky galaxies
    morphology_columns = [
       "redshift",
       "spheroidEllipticity1",
       "spheroidEllipticity2",
       "spheroidHalfLightRadiusArcsec",
       "diskEllipticity1",
       "diskEllipticity2",
       "diskHalfLightRadiusArcsec",
       "um_source_galaxy_obs_sm",
    ]
    spectral_columns = ALL_DIFFSKY_PNAMES
    columns = list(set(morphology_columns + spectral_columns))

    def __init__(self, data):
        self.data = data
        self.num_rows = utils.count_rows(self.data)

    # def to_iter(self, *args, **kwargs):
    #     def _iterator(self, *args, **kwargs):
    #         for i in range(self.num_rows):
    #             yield self.get_params(i, *args, **kwargs)

    #     return _iterator(self, *args, **kwargs)

    def get_morphology_params(self, i, knots=False):
        redshift = utils.unwrap(self.data["redshift"][i])
        spheroidEllipticity1 = utils.unwrap(self.data["spheroidEllipticity1"][i])
        spheroidEllipticity2 = utils.unwrap(self.data["spheroidEllipticity2"][i])
        spheroidHalfLightRadiusArcsec = utils.unwrap(self.data["spheroidHalfLightRadiusArcsec"][i])
        diskEllipticity1 = utils.unwrap(self.data["diskEllipticity1"][i])
        diskEllipticity2 = utils.unwrap(self.data["diskEllipticity2"][i])
        diskHalfLightRadiusArcsec = utils.unwrap(self.data["diskHalfLightRadiusArcsec"][i])
        diffsky_param_data = _get_diffsky_params(self.data, i)

        if knots:
            n_knots = _get_knot_n(self.data, i)
        else:
            n_knots = 0

        return (
            redshift,
            spheroidEllipticity1,
            spheroidEllipticity2,
            spheroidHalfLightRadiusArcsec,
            diskEllipticity1,
            diskEllipticity2,
            diskHalfLightRadiusArcsec,
            diffsky_param_data,
            n_knots,
        )

    def get_spectrum_params(self, i):
        redshift = utils.unwrap(self.data["redshift"][i])
        diffsky_param_data = _get_diffsky_params(self.data, i)
        return (
            redshift,
            diffsky_param_data,
        )

    def get_params(self, i, knots=False):
        redshift = utils.unwrap(self.data["redshift"][i])
        spheroidEllipticity1 = utils.unwrap(self.data["spheroidEllipticity1"][i])
        spheroidEllipticity2 = utils.unwrap(self.data["spheroidEllipticity2"][i])
        spheroidHalfLightRadiusArcsec = utils.unwrap(self.data["spheroidHalfLightRadiusArcsec"][i])
        diskEllipticity1 = utils.unwrap(self.data["diskEllipticity1"][i])
        diskEllipticity2 = utils.unwrap(self.data["diskEllipticity2"][i])
        diskHalfLightRadiusArcsec = utils.unwrap(self.data["diskHalfLightRadiusArcsec"][i])

        diffsky_param_data = _get_diffsky_params(self.data, i)

        if knots:
            n_knots = _get_knot_n(self.data, i)
        else:
            n_knots = 0

        return (
            redshift,
            spheroidEllipticity1,
            spheroidEllipticity2,
            spheroidHalfLightRadiusArcsec,
            diskEllipticity1,
            diskEllipticity2,
            diskHalfLightRadiusArcsec,
            diffsky_param_data,
            n_knots,
        )

