import galsim


class Galaxies:
    def __init__(self):
        pass

    def get_morphology(self):
        raise NotImplementedError

    def get_spectrum(self):
        raise NotImplementedError

    def get_galaxy(self):
        raise NotImplementedError
