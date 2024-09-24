from . import constants

from astropy import units as u


class AirMass:
    def kasten_young(altitude: u.Quantity,
                     elevation: u.Quantity = 0 * u.m):
        return np.where(
            altitude >= 0 * u.m,
            air_density(elevation) / air_density(0) / (np.sin(altitude)) + 0.50572 * ((altitude.degree + 6.07995) ** (-1.6364)),
            np.inf
        )

    def pickering2002(altitude: u.Quantity,
                      elevation: u.Quantity = 0 * u.m):
        return np.where(
            altitude >= 0 * u.m,
            air_density(elevation) / air_density(0) / (np.sin(altitude + np.radians(244 / (165 + 47 * altitude.degree ** 1.1)))),
            np.inf
        )

    def attenuate(flux: np.ndarray,
                  air_mass: np.ndarray,
                  one: float = constants.AttenuationOneAirMass):
        return np.where(
            air_mass <= 100,
            flux * np.exp(one * air_mass),
            0
        )

