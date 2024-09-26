import pytest
import datetime

from astropy.coordinates import EarthLocation
from astropy import units as u
from astropy.time import Time

from amosutils.catalogue import Catalogue


@pytest.fixture
def hyg30():
    return Catalogue('tests/HYG30.tsv')

@pytest.fixture
def modra():
    return EarthLocation(17.27 * u.deg, 48.37 * u.deg, 531 * u.m)


class TestCatalogue:
    def test_load(self, hyg30):
        assert isinstance(hyg30, Catalogue)

    def test_polaris(self, hyg30, modra):
        # Check that latitude of Polaris is within 1 degree of the observer's latitude
        altaz = hyg30.altaz(modra)
        assert altaz[47].alt.degree == pytest.approx(modra.lat.degree, abs=1)

    def test_sirius(self, hyg30, modra):
        altaz = hyg30.altaz(modra, Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC)))
        assert altaz[0].alt.degree == pytest.approx(-25.6, abs=0.2)
        assert altaz[0].az.degree == pytest.approx(86.6, abs=0.2)
