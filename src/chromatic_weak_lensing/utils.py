from collections import OrderedDict, UserDict
import logging
import math

import astropy.units as u
from astropy.constants import G, sigma_sb
import h5py
import galsim
import numpy as np
import pyarrow as pa
from dust_extinction.parameter_averages import G23


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
        logL=None,
        L=None,
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
        self._logL = logL
        self._L = L
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

        if (self.logL is not None) and (self.L is not None):
            assert math.isclose(self.logL, math.log(self.L, 10))
        elif (self.logL is not None) and (self.L is None):
            self._L = 10 ** self._logL
        elif (self.logL is None) and (self.L is not None):
            self._logL = math.log(self._L, 10)

        if (self.distance_modulus is not None) and (self.distance is not None):
            assert math.isclose(
                self.distance_modulus,
                get_distance_modulus(self.distance),
            )
        elif (self.distance_modulus is None) and (self.distance is not None):
            self._distance_modulus = get_distance_modulus(self.distance)
        elif (self.distance is None) and (self.distance_modulus is not None):
            self._distance = get_distance(self.distance_modulus)

        # Newton
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

        # Stefan-Boltzmann
        if (self.logL is None) and (self.radius is not None) and (self.logT is not None):
            # L = 4 pi sigma R^2 T^4 [L_sun]
            _L = (
                4 * math.pi * sigma_sb * (self._radius * u.R_sun)**2 * (self._T * u.K)**4
            ).decompose().value
            self._logL = math.log(_L, 10)
            self._L = _L
        elif (self.logL is not None) and (self.radius is not None) and (self.logT is None):
            # T = (L / 4 pi sigma R^2)^(1/4) [K]
            _T = (
                (self._L * u.L_sun / (4 * math.pi * sigma_sb * (self._radius * u.R_sun)**2))**(1/4)
            ).decompose().value
            self._logT = math.log(_T, 10)
            self._T = _T
        elif (self.logL is not None) and (self.radius is None) and (self.logT is not None):
            # R = (L / 4 pi sigma T^4)^(1/2) [R_sun]
            self._radius = (
                (self._L * u.L_sun / (4 * math.pi * sigma_sb * (self._T * u.K)**4))**(1/2)
            ).decompose().value

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
    def logL(self):
        return self._logL

    @property
    def L(self):
        return self._L

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
            f"    logL: {self.logL}\n"
            f"    z: {self.z}\n"
            f"    phase: {self.phase}\n"
            f"    composition: {self.composition}\n"
            f"    distance_modulus: {self.distance_modulus}\n"
        )



def unwrap(value):
    if isinstance(value, pa.Scalar):
        return value.as_py()
    else:
        return value


def count_rows(data):
    match data:
        case dict() | OrderedDict() | UserDict():
            return len(next(iter(data.values())))
        case h5py.Group():
            return len(next(iter(data.values())))
        case np.ndarray():
            return len(data)
        case pa.Table() | pa.RecordBatch():
            return data.num_rows
        case _:
            return None


def get_distance_modulus(distance):
    return 5 * math.log(distance, 10) - 5


def get_distance(distance_modulus):
    return 10**(1 + distance_modulus / 5)


def get_surface_area(distance_modulus):
    """
    Area of the sphere whose radius extends from the source to the observer
    """
    distance = get_distance(distance_modulus)
    return 4 * math.pi * distance**2


def get_extinction(av, Rv=3.1, red_limit=12_000):
    ext = G23(Rv=Rv)

    eps = 1e-7  # machine epsilon
    wl_min = 1e4 / ext.x_range[1]  + eps # 1/micron to angstrom
    wl_max = 1e4 / ext.x_range[0]  - eps # 1/micron to angstrom
    if wl_max > red_limit:
        wl_max = red_limit

    n_wl = int(wl_max - wl_min) // 10
    wls = np.linspace(wl_min, wl_max, n_wl)

    extinction = ext.extinguish(wls * u.AA, Av=av)

    extinction_table = galsim.LookupTable(
        wls,
        extinction,
    )
    extinction_sed = galsim.SED(
        extinction_table,
        wave_type="angstrom",
        flux_type="1",
    )

    return extinction_sed
