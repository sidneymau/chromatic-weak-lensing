import logging
import math
import time

import galsim

logger = logging.getLogger(__name__)


class Stars:
    def __init__(self):
        self.name = None

    def get_morphology(self):
        return galsim.DeltaFunction()

    def get_spectrum(self):
        raise NotImplementedError

    def get_star(self, *args, **kwargs):
        spectrum = self.get_spectrum(*args, **kwargs)
        morphology = self.get_morphology()

        return morphology * spectrum
