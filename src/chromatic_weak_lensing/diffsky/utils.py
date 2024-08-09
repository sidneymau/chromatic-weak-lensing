import functools
import os
import urllib.request

from lsstdesc_diffsky import read_diffskypop_params
from lsstdesc_diffsky.legacy.roman_rubin_2023.dsps.data_loaders.load_ssp_data import load_ssp_templates_singlemet

DSPS_TEST_DATA_URL = "https://portal.nersc.gov/project/hacc/aphearin/lsstdesc_diffsky_data/roman_rubin_2023_z_0_1_cutout_9043.testdata.hdf5"
DSPS_TEST_DATA = "diffsky.testdata.hdf5"
DSPS_SSP_DATA_URL = "https://portal.nersc.gov/project/hacc/aphearin/DSPS_data/ssp_data_fsps_v3.2_age.h5"
DSPS_SSP_DATA = "dsps_ssp_data_singlemet.h5"

MOCK_NAME = "roman_rubin_2023"


def retrieve_dsps_test_data():
    print(f"Retrieving {DSPS_TEST_DATA_URL} > {DSPS_TEST_DATA}")
    status = urllib.request.urlretrieve(DSPS_TEST_DATA_URL, DSPS_TEST_DATA)
    return status


def retrieve_dsps_ssp_data():
    print(f"Retrieving {DSPS_SSP_DATA_URL} > {DSPS_SSP_DATA}")
    status = urllib.request.urlretrieve(DSPS_SSP_DATA_URL, DSPS_SSP_DATA)
    return status


@functools.cache
def load_dsps_ssp_data():
    if not os.path.exists(DSPS_SSP_DATA):
        retrieve_dsps_ssp_data()
    data = load_ssp_templates_singlemet(DSPS_SSP_DATA)
    return data


@functools.cache
def load_diffskypop_params():
    return read_diffskypop_params(MOCK_NAME)

