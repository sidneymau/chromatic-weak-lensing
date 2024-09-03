import functools
import logging
import os
import urllib.request
import warnings

from lsstdesc_diffsky import read_diffskypop_params
from lsstdesc_diffsky.legacy.roman_rubin_2023.dsps.data_loaders.load_ssp_data import load_ssp_templates_singlemet


logger = logging.getLogger(__name__)


# DSPS_TEST_DATA_URL = "https://portal.nersc.gov/project/hacc/aphearin/lsstdesc_diffsky_data/roman_rubin_2023_z_0_1_cutout_9043.testdata.hdf5"
# DSPS_SSP_DATA_URL = "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_age.h5"
DSPS_SSP_DATA = "dsps_ssp_data_singlemet.h5"

MOCK_NAME = "roman_rubin_2023"


def retrieve_dsps_ssp_data(fname):
    logger.info(f"Retrieving {DSPS_SSP_DATA_URL} > {fname}")
    status = urllib.request.urlretrieve(DSPS_SSP_DATA_URL, fname)
    return status


@functools.cache
def load_dsps_ssp_data():
    dsps_ssp_data = os.path.join(
        os.path.dirname(__file__),
        "data",
        DSPS_SSP_DATA,
    )

    logger.info(f"loading DSPS_SSP_DATA from {dsps_ssp_data}")
    data = load_ssp_templates_singlemet(dsps_ssp_data)
    return data


@functools.cache
def load_diffskypop_params():
    logger.info(f"reading diffsky population parameters for {MOCK_NAME}")
    return read_diffskypop_params(MOCK_NAME)

