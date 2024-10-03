import datetime
import numpy as np
import pandas as pd

from pathlib import Path

from astropy.coordinates import EarthLocation, FK5, SkyCoord, AltAz, get_body, concatenate
from astropy.time import Time
from astropy import units as u


class Catalogue:
    def __init__(self, filename: Path, *, planets: bool = True):
        # Load stars from catalogue
        self.stars = pd.read_csv(filename, sep='\t', header=1)
        self.stars_skycoord = SkyCoord(self.stars.ra.to_numpy() * u.deg,
                                       self.stars.dec.to_numpy() * u.deg,
                                       frame=FK5(equinox=Time('J2000')))

    def build_planets(self, location: EarthLocation, time: Time = None):
        if time is None:
            time = Time(datetime.datetime.now(tz=datetime.UTC))

        earth = get_body('earth', time)

        index = len(self.stars)
        planets = pd.DataFrame(columns=['ra', 'dec', 'dist', 'vmag', 'absmag'])

        for name in ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']:
            body = get_body(name, time=time, location=location)
            sundist = body.hcrs.distance
            phase = earth.hcrs.separation(body.hcrs)
            planets = pd.concat([planets, pd.DataFrame(
                data=[
                    [
                        body.ra.degree,
                        body.dec.degree,
                        body.distance.to(u.lightyear).value,
                        self.brightness(name, body.distance, sundist, phase),
                        -10
                    ]
                ],
                index=[index])])
            index += 1

        self.planets_skycoord = SkyCoord(planets.ra.to_numpy() * u.deg,
                                         planets.dec.to_numpy() * u.deg,
                                         frame=FK5(equinox=Time('J2000')))

    def altaz(self, location: EarthLocation, time: Time = None, *, planets: bool = True):
        if time is None:
            time = Time(datetime.datetime.now(tz=datetime.UTC))

        altaz = AltAz(location=location, obstime=time, pressure=100000 * u.pascal, obswl=550 * u.nm)
        if planets:
            self.build_planets(location, time)
            total = concatenate([self.stars_skycoord, self.planets_skycoord])
            return total.transform_to(altaz)
        else:
            return self.stars_skycoord.transform_to(altaz)

    @staticmethod
    def brightness(planet: str, distance_earth: float, distance_sun: float, phase: u.Quantity):
        """
            Get approximate visual magnitude of a planet.
            Stolen from APC, Montenbruck 1999
        """
        p = phase.degree / 100.0

        match planet:
            case 'mercury':
                mag = -0.42 + (3.80 - (2.73 - 2 * p) * p) * p
            case 'venus':
                mag = -4.40 + (0.09 + (2.39 - 0.65 * p) * p) * p
            case 'mars':
                mag = -1.52 + 1.6 * p
            case 'jupiter':
                mag = -9.4 + 0.5 * p
            case 'saturn':
                # Currently we do not care about the rings
                sd = 0 # np.abs(np.sin(lat))
                dl = 0 # np.abs((dlong + np.pi) % (2 * np.pi) - np.pi) / 100
                mag = -8.88 + 2.60 * sd + 1.25 * sd**2 + 4.4 * dl
            case 'uranus':
                mag = -7.19
            case 'neptune':
                mag = -6.87

        return mag + 5 * np.log10(distance_earth.to(u.au).value * distance_sun.to(u.au).value)