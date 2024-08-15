import logging
import math
import time

import astropy.units as u
from astropy.constants import G
import galsim

from chromatic_weak_lensing import utils


logger = logging.getLogger(__name__)


class StellarParams:
    def __init__(
        self,
        ra=None,
        dec=None,
        Av=None,
        mass=None,
        radius=None,
        logT=None,
        T=None,
        logg=None,
        g=None,
        logl=None,
        l=None,
        z=None,
        distance_modulus=None,
        distance=None,
        phase=None,
        composition=None,
    ):
        self._ra = ra
        self._dec = dec
        self._Av = Av
        self._mass = mass
        self._radius = radius
        self._logg = logg
        self._g = g
        self._logT = logT
        self._T = T
        self._logl = logl
        self._l = l
        self._distance_modulus = distance_modulus
        self._distance = distance
        self._z = z
        self._phase = phase
        self._composition = composition

        if (self.logg is not None) and (self.g is not None):
            assert math.isclose(self.logg, math.log(self.g, 10))
        elif (self.logg is not None) and (self.g is None):
            self._g = 10 ** self._logg
        elif (self.logg is None) and (self.g is not None):
            self._logg = math.log(self._g, 10)

        if (self.logT is not None) and (self.T is not None):
            assert math.isclose(self.logT, math.log(self.T, 10))
        elif (self.logT is not None) and (self.T is None):
            self._T = 10 ** self._logT
        elif (self.logT is None) and (self.T is not None):
            self._logT = math.log(self._T, 10)

        if (self.logl is not None) and (self.l is not None):
            assert math.isclose(self.logl, math.log(self.l, 10))
        elif (self.logl is not None) and (self.l is None):
            self._l = 10 ** self._logl
        elif (self.logl is None) and (self.l is not None):
            self._logl = math.log(self._l, 10)

        if (self.distance_modulus is not None) and (self.distance is not None):
            assert math.isclose(
                self.distance_modulus,
                utils.get_distance_modulus(self.distance),
            )
        elif (self.distance_modulus is None) and (self.distance is not None):
            self._distance_modulus = utils.get_distance_modulus(self.distance)
        elif (self.distance is None) and (self.distance_modulus is not None):
            self._distance = utils.get_distance(self.distance_modulus)

        if (self.mass is not None) and (self.radius is not None) and (self.logg is None):
            # g = G M / R^2 [cm / s^2]
            _g = (
                G * (self._mass * u.M_sun) / (self._radius * u.R_sun)**2
            ).cgs.value
            self._logg = math.log(_g, 10)
            self._g = _g
        elif (self.mass is not None) and (self.radius is None) and (self.logg is not None):
            # R = (G M / g)^(1/2) [R_sun]
            self._radius = (
                (G * (self._mass * u.M_sun) / (self._g * u.cm / u.s**2))**(1/2) / u.R_sun
            ).decompose()
        elif (self.mass is None) and (self.radius is not None) and (self.logg is not None):
            # M = g R^2 / G [M_sun]
            self._mass = (
                (self._g * u.cm / u.s**2) * (self._radius * u.R_sun) / G
            ).decompose()

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec

    @property
    def Av(self):
        return self._Av

    @property
    def mass(self):
        return self._mass

    @property
    def radius(self):
        return self._radius

    @property
    def logg(self):
        return self._logg

    @property
    def g(self):
        return self._g

    @property
    def logT(self):
        return self._logT

    @property
    def T(self):
        return self._T

    @property
    def logl(self):
        return self._logl

    @property
    def l(self):
        return self._l

    @property
    def distance(self):
        return self._distance

    @property
    def distance_modulus(self):
        return self._distance_modulus

    @property
    def z(self):
        return self._z

    @property
    def phase(self):
        return self._phase

    @property
    def composition(self):
        return self._composition

    def __repr__(self):
        return (
            f"Stellar Parameters\n"
            f"    ra: {self.ra}\n"
            f"    dec: {self.dec}\n"
            f"    Av: {self.Av}\n"
            f"    mass: {self.mass}\n"
            f"    radius: {self.radius}\n"
            f"    logg: {self.logg}\n"
            f"    logT: {self.logT}\n"
            f"    logl: {self.logl}\n"
            f"    z: {self.z}\n"
            f"    phase: {self.phase}\n"
            f"    composition: {self.composition}\n"
            f"    distance_modulus: {self.distance_modulus}\n"
        )


class Stars:
    def __init__(self):
        self.name = None

    def get_params(self):
        raise NotImplementedError

    def get_morphology(self):
        return galsim.DeltaFunction()

    def get_spectrum(self):
        raise NotImplementedError

    def get_star(self, *args, **kwargs):
        spectrum = self.get_spectrum(*args, **kwargs)
        morphology = self.get_morphology()

        return morphology * spectrum
