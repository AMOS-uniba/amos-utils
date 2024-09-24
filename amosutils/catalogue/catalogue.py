import pandas as pd

from pathlib import Path

from astropy.coordinates import EarthLocation, FK5, SkyCoord, AltAz, get_body
from astropy.time import Time
from astropy import units as u


class Catalogue:
    def __init__(self, filename: Path, *, planets: bool = True):
        # Load stars from catalogue
        self.df = pd.read_csv(filename, sep='\t')
        self.skycoord = SkyCoord(self.df.ra.to_numpy * u.deg,
                                 self.df.dec.to_numpy() * u.deg,
                                 frame = FK5(equinox=Time('J2000')))

    def altaz(self, location: EarthLocation, time: Time = Time.now):
        altaz = AltAz(location=location, obstime=time, pressure=100000 * u.pascal, obswl=550 * u.nm)
        return self.skycoord.transform_to(altaz)




