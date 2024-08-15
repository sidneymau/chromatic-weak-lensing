import logging
import math
import os
import time

import astropy.units as u
import galsim
import numpy as np

from dl import queryClient as qc

from chromatic_weak_lensing import utils
from chromatic_weak_lensing import StellarParams


logger = logging.getLogger(__name__)


def _query_data_lab(ring, N):
    """Fetch TRILEGAL stars for a given ra/dec region."""
    query = \
       """SELECT
              mass, ra, dec, logage, logg, logte, logl, z, mu0, av, label, c_o,
              umag, gmag, rmag, imag, zmag, ymag
          FROM lsst_sim.simdr2
          WHERE (ring256={}) AND label=1
          LIMIT {}
       """.format(ring, N)

    logger.info(f"Submitting query:\n{query}")

    result = qc.query(sql=query, fmt="structarray")

    return result


class LSST_Sim:
    def __init__(self, data):
        self.data = data
        self.num_rows = utils.count_rows(data)

    def get_params(self, i):
        sparams = StellarParams(
            ra=self.data["ra"][i],
            dec=self.data["dec"][i],
            Av=self.data["av"][i],
            mass=self.data["mass"][i],
            logg=self.data["logg"][i],
            logT=self.data["logte"][i],
            logL=self.data["logl"][i],
            distance_modulus=self.data["mu0"][i],
            z=self.data["z"][i],
            phase=self.data["label"][i],
            composition=self.data["c_o"][i],
        )
        return sparams
