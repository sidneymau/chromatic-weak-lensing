import logging
import time

from chromatic_weak_lensing import Blackbody, MainSequence, Stars


logger = logging.getLogger(__name__)


class SimpleStar(Stars):
    def __init__(self, *args, **kwargs):
        self.name = "SimpleStar"
        self.spectrum = Blackbody()
        self.stellar_mass = 1
        self.stellar_params = MainSequence.get_params(self.stellar_mass)

    def get_params(self, *args, **kwargs):
        return self.spectrum.get_params(self.stellar_params)

    def get_spectrum(self, *args, **kwargs):
        _blackbody_params = self.get_params()
        return self.spectrum.get_spectrum(*_blackbody_params)
