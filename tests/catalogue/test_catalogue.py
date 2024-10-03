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
def ago():
    return EarthLocation(17.27 * u.deg, 48.37 * u.deg, 531 * u.m)

@pytest.fixture
def basic(hyg30, ago):
    return hyg30.altaz(ago, Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC)))


class TestCatalogue:
    def test_load(self, hyg30):
        assert isinstance(hyg30, Catalogue)

    def test_with_planets(self, hyg30, ago):
        assert len(hyg30.altaz(ago)) == 5075

    def test_polaris(self, hyg30, ago):
        # Check that latitude of Polaris is within 1 degree of the observer's latitude
        altaz = hyg30.altaz(ago)
        assert altaz[47].alt.degree == pytest.approx(ago.lat.degree, abs=1)

    def test_sirius(self, hyg30, ago):
        altaz = hyg30.altaz(ago, Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC)))
        assert altaz[0].alt.degree == pytest.approx(-25.6, abs=0.2)
        assert altaz[0].az.degree == pytest.approx(86.6, abs=0.2)

    def test_planets(self, hyg30, ago):
        hyg30.build_planets(ago)
        assert len(hyg30.planets_skycoord) == 7

    def test_jupiter(self, hyg30, ago):
        pass
        #assert hyg30.vmag[5071] == pytest.approx(-2, abs=0.5)

