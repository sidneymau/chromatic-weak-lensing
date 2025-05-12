import logging
import time

from chromatic_weak_lensing import Blackbody, MainSequence, Stars


logger = logging.getLogger(__name__)


class SimpleStar(Stars):
    def __init__(self, *args, **kwargs):
        self.name = "SimpleStar"
        self.spectrum = Blackbody()
        self.stellar_mass = 1

    def get_spectrum(self, *args, **kwargs):
        _params = MainSequence.get_params(self.stellar_mass)
        _blackbody_params = self.spectrum.get_params(_params)
        return self.spectrum.get_spectrum(*_blackbody_params)
