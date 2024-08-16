import logging
import math

from scipy.interpolate import interp1d

from chromatic_weak_lensing import StellarParams


logger = logging.getLogger(__name__)


class MainSequence:
    # logM, logL, logR
    # From Allen, C. W., Astrophysical Quantities, Athlone Press, 1973,
    # via Martin V. Zombeck's Handbook of Space Astronomy & Astrophysics
    # accessed at https://ads.harvard.edu/cgi-bin/bbrowse?book=hsaa&page=72

    logM = [
        -1.0,
        -0.8,
        -0.6,
        -0.4,
        -0.2,
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
    ]

    logL = [
        -2.9,
        -2.5,
        -2.0,
        -1.5,
        -0.8,
        0.0,
        0.8,
        1.6,
        2.3,
        3.0,
        3.7,
        4.4,
        4.9,
        5.4,
        6.0,
    ]

    logR = [
        -0.9,
        -0.7,
        -0.5,
        -0.3,
        -0.14,
        0.00,
        0.10,
        0.32,
        0.49,
        0.58,
        0.72,
        0.86,
        1.00,
        1.15,
        1.3,
    ]

    interpolator = interp1d(logM, (logL, logR))

    phase = 1
    composition = 1

    # solar metallicity
    # z = 0.019  # https://arxiv.org/abs/0809.4261
    z = 0.0134  # https://arxiv.org/abs/0909.0948

    @classmethod
    def get_params(self, mass, distance=None, distance_modulus=None):
        if (distance is None) and (distance_modulus is None):
            # distance to typical star in MW
            distance_modulus = 14

        logM = math.log(mass, 10)

        logL, logR = self.interpolator(logM)
        radius = 10 ** logR

        return StellarParams(
            mass=mass,
            logL=logL,
            radius=radius,
            phase=self.phase,
            composition=self.composition,
            z=self.z,
            distance=distance,
            distance_modulus=distance_modulus,
        )
