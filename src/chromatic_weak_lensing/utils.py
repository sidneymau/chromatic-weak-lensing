import logging
import math

import pyarrow as pa

logger = logging.getLogger(__name__)


def get_surface_area(distance_modulus):
    """
    Area of the sphere whose radius extends from the source to the observer
    """
    distance = 10**(1 + distance_modulus / 5)
    return 4 * math.pi * distance**2


def unwrap(value):
    if isinstance(value, pa.Scalar):
       return value.as_py()
    else:
       return value
