import functools
import logging
import time

import astropy.units as u
from astropy.constants import h, c, k_B
import math
import galsim
import numpy as np

from chromatic_weak_lensing import utils
from chromatic_weak_lensing import Blackbody, Galaxies, MainSequence


logger = logging.getLogger(__name__)


class SimpleGalaxy(Galaxies):
    def __init__(self, *args, **kwargs):
        self.name = "SimpleGalaxy"
        self.spectrum = Blackbody()
        self.stellar_mass = 1
        self.half_light_radius = 0.3

    def get_morphology(self, *args, **kwargs):
        return galsim.Exponential(half_light_radius=self.half_light_radius)

    def get_spectrum(self, *args, **kwargs):
        _params = MainSequence.get_params(self.stellar_mass)
        _blackbody_params = self.spectrum.get_params(_params)
        return self.spectrum.get_spectrum(*_blackbody_params)

    def get_galaxy(self, *args, **kwargs):
        spectrum = self.get_spectrum()
        morphology = self.get_morphology()

        return morphology * spectrum
