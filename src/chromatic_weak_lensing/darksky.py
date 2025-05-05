import logging
import os

import galsim

logger = logging.getLogger(__name__)

# darksky.dat from https://raw.githubusercontent.com/lsst-pst/syseng_throughputs/main/siteProperties/darksky.dat


class Darksky:
    filename = "darksky.dat"

    def __init__(self):
        self.name = "darksky"
        self.sed = self._load_darksky_SED(self.filename)

    def __call__(self, *args, **kwargs):
        return self.sed(*args, **kwargs)

    def _load_darksky_SED(self, filename):
        darksky_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            filename,
        )
        logger.info(f"loading darksky from {darksky_path}")
        darksky = galsim.SED(darksky_path, wave_type="nm", flux_type="flambda")
        return darksky

    def get_noise_sigma(self, throughput, npixel, ncoadd=1):
        # get background noise according to a dark sky spectrum
        _flux = self.sed.calculateFlux(throughput)

        # need standard deviation of counts in each pixel
        # for a Poisson distribution, the mean and variance are equal, so
        # we suppose the standard deviation on counts is the root of the
        # total flux; then we divide by the number of pixels
        # we also divide the flux by the number of coadds to get the
        # right reduction relative to the galaxies
        # equivalently, we could multiply both fluxes by n
        _flux_per_pixel = (_flux / ncoadd) ** (1/2) / npixel

        logger.info(f"computed noise standard deviation {_flux_per_pixel} flux per pixel")

        return _flux_per_pixel
