"""
Comprehensive tests for the vanHOzone package.

Tests the van Heuklon (1979) Ozone model implementation for estimating
atmospheric ozone concentration.
"""
import pytest
import numpy as np
from datetime import datetime
from vanHOzone import get_ozone_conc


# Test markers for grouping
pytestmark = pytest.mark.vanhozone


class TestScalarInputs:
    """Test cases for scalar (single value) inputs."""

    def test_scalar_northern_hemisphere(self):
        """Test scalar values in Northern hemisphere (London)."""
        result = get_ozone_conc(51.5, -0.1, "2015-06-15")
        assert result == pytest.approx([383.85882587], rel=1e-7)

    def test_scalar_southern_hemisphere(self):
        """Test scalar values in Southern hemisphere (Sydney)."""
        result = get_ozone_conc(-33.9, 151.2, "2015-12-15")
        assert result == pytest.approx([313.50199155], rel=1e-7)

    def test_scalar_equator(self):
        """Test scalar values at the Equator (Quito)."""
        result = get_ozone_conc(0.0, -78.5, "2015-03-21")
        assert result == pytest.approx([235.0], rel=1e-7)

    def test_datetime_object(self):
        """Test using a datetime object instead of string."""
        dt = datetime(2015, 9, 23, 12, 0, 0)
        result = get_ozone_conc(40.7, -74.0, dt)
        assert result == pytest.approx([316.87971504], rel=1e-7)


class TestArrayInputs:
    """Test cases for array inputs."""

    def test_array_inputs_same_timestamp(self):
        """Test array inputs with same timestamp for all locations."""
        lats = [51.5, -33.9, 0.0, 40.7, -23.5]
        lons = [-0.1, 151.2, -78.5, -74.0, -46.6]
        result = get_ozone_conc(lats, lons, "2015-06-21")
        expected = [381.33586549, 289.17339348, 235.0, 352.69969946, 267.86515712]
        assert result == pytest.approx(expected, rel=1e-7)

    def test_array_inputs_array_timestamps(self):
        """Test array inputs with array of timestamps."""
        lats = [51.5, -33.9, 0.0]
        lons = [-0.1, 151.2, -78.5]
        timestamps = ["2015-01-15", "2015-07-15", "2015-12-15"]
        result = get_ozone_conc(lats, lons, timestamps)
        expected = [351.4262453, 295.71418863, 235.0]
        assert result == pytest.approx(expected, rel=1e-7)

    def test_numpy_arrays_as_input(self):
        """Test that numpy arrays work as inputs."""
        lats = np.array([51.5, -33.9, 0.0])
        lons = np.array([-0.1, 151.2, -78.5])
        result = get_ozone_conc(lats, lons, "2015-06-21")
        expected = [381.33586549, 289.17339348, 235.0]
        assert result == pytest.approx(expected, rel=1e-7)

    def test_datetime_array(self):
        """Test with array of datetime objects."""
        lats = [51.5, -33.9, 0.0]
        lons = [-0.1, 151.2, -78.5]
        timestamps = [
            datetime(2015, 1, 15),
            datetime(2015, 7, 15),
            datetime(2015, 12, 15)
        ]
        result = get_ozone_conc(lats, lons, timestamps)
        expected = [351.4262453, 295.71418863, 235.0]
        assert result == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize("lat,lon,date,expected", [
    (40.7, -74.0, "2015-01-15", 330.35930121),  # Winter
    (40.7, -74.0, "2015-04-15", 360.65695415),  # Spring
    (40.7, -74.0, "2015-07-15", 343.68467842),  # Summer
    (40.7, -74.0, "2015-10-15", 312.72004167),  # Fall
])
class TestNorthernHemisphereSeasons:
    """Test Northern hemisphere seasonal variations (New York)."""

    def test_season(self, lat, lon, date, expected):
        """Test ozone concentration for different seasons."""
        result = get_ozone_conc(lat, lon, date)
        assert result == pytest.approx([expected], rel=1e-7)


@pytest.mark.parametrize("lat,lon,date,expected", [
    (-37.8, 144.9, "2015-01-15", 319.17149283),  # Summer
    (-37.8, 144.9, "2015-04-15", 293.71526119),  # Fall
    (-37.8, 144.9, "2015-07-15", 308.07017292),  # Winter
    (-37.8, 144.9, "2015-10-15", 334.08756962),  # Spring
])
class TestSouthernHemisphereSeasons:
    """Test Southern hemisphere seasonal variations (Melbourne)."""

    def test_season(self, lat, lon, date, expected):
        """Test ozone concentration for different seasons."""
        result = get_ozone_conc(lat, lon, date)
        assert result == pytest.approx([expected], rel=1e-7)


class TestExtremeLatitudes:
    """Test cases for extreme latitudes."""

    @pytest.mark.parametrize("lat,location,expected", [
        (80, "Arctic", 402.56685809),
        (-80, "Antarctic", 288.11037702),
    ])
    def test_extreme_latitudes(self, lat, location, expected):
        """Test extreme latitude calculations."""
        result = get_ozone_conc(lat, 0.0, "2015-06-21")
        assert result == pytest.approx([expected], rel=1e-7)


class TestLongitudeVariations:
    """Test cases for different longitudes."""

    @pytest.mark.parametrize("lon,direction,expected", [
        (-120, "West", 362.38905981),
        (0, "Prime Meridian", 362.38905981),
        (120, "East", 374.73667068),
    ])
    def test_different_longitudes(self, lon, direction, expected):
        """Test Northern hemisphere with different longitudes."""
        result = get_ozone_conc(45.0, lon, "2015-06-15")
        assert result == pytest.approx([expected], rel=1e-7)


class TestDateFormats:
    """Test various date format parsing."""

    def test_iso8601_datetime_format(self):
        """Test ISO 8601 datetime format with time component."""
        result = get_ozone_conc(35.0, 139.7, "2015-08-30T14:30:00")
        assert result == pytest.approx([308.48281735], rel=1e-7)

    @pytest.mark.parametrize("date_str,format_name", [
        ("2015-05-20", "ISO (YYYY-MM-DD)"),
        ("05/20/2015", "US (MM/DD/YYYY)"),
        ("20/05/2015", "European (DD/MM/YYYY)"),
    ])
    def test_date_formats(self, date_str, format_name):
        """Test various date formats all parse to same result."""
        result = get_ozone_conc(52.5, 13.4, date_str)
        assert result == pytest.approx([411.43655659], rel=1e-7)


class TestErrorHandling:
    """Test error conditions and edge cases."""

    def test_mismatched_lat_lon_lengths(self):
        """Test that mismatched lat and lon arrays raise ValueError."""
        lats = [51.5, -33.9, 0.0]
        lons = [-0.1, 151.2]  # Different length

        with pytest.raises(ValueError, match="lan and lon"):
            get_ozone_conc(lats, lons, "2015-06-21")

    def test_mismatched_timestamp_length(self):
        """Test that mismatched timestamp array length raises an error."""
        lats = [51.5, -33.9, 0.0]
        lons = [-0.1, 151.2, -78.5]
        timestamps = ["2015-01-15", "2015-07-15"]  # Different length

        # Due to the exception handling in the code, this raises TypeError
        with pytest.raises(TypeError):
            get_ozone_conc(lats, lons, timestamps)


class TestReturnValues:
    """Test return value properties."""

    def test_result_is_numpy_array(self):
        """Test that result is always a numpy array."""
        result = get_ozone_conc(51.5, -0.1, "2015-06-15")
        assert isinstance(result, np.ndarray)

    def test_result_shape_scalar(self):
        """Test that scalar inputs return array of shape (1,)."""
        result = get_ozone_conc(51.5, -0.1, "2015-06-15")
        assert result.shape == (1,)

    def test_result_shape_array(self):
        """Test that array inputs return array of correct shape."""
        lats = [51.5, -33.9, 0.0, 40.7, -23.5]
        lons = [-0.1, 151.2, -78.5, -74.0, -46.6]
        result = get_ozone_conc(lats, lons, "2015-06-21")
        assert result.shape == (5,)

    def test_positive_ozone_values(self):
        """Test that ozone concentration values are always positive."""
        lats = [80, 45, 0, -45, -80]
        lons = [0, 90, -90, 180, -180]
        result = get_ozone_conc(lats, lons, "2015-06-21")
        assert np.all(result > 0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_leap_year_date(self):
        """Test with a leap year date."""
        result = get_ozone_conc(51.5, -0.1, "2016-02-29")
        assert isinstance(result, np.ndarray)
        assert result[0] > 0

    @pytest.mark.parametrize("date", [
        "2015-01-01",  # New Year's Day
        "2015-12-31",  # New Year's Eve
    ])
    def test_year_boundaries(self, date):
        """Test dates at year boundaries."""
        result = get_ozone_conc(40.7, -74.0, date)
        assert result[0] > 0


# Test data fixtures for reuse
@pytest.fixture
def sample_locations():
    """Fixture providing sample geographic locations."""
    return {
        'london': (51.5, -0.1),
        'sydney': (-33.9, 151.2),
        'new_york': (40.7, -74.0),
        'melbourne': (-37.8, 144.9),
        'tokyo': (35.0, 139.7),
    }


@pytest.fixture
def sample_dates():
    """Fixture providing sample dates for testing."""
    return {
        'winter_nh': "2015-01-15",
        'spring_nh': "2015-04-15",
        'summer_nh': "2015-07-15",
        'fall_nh': "2015-10-15",
        'equinox_spring': "2015-03-21",
        'equinox_fall': "2015-09-23",
        'solstice_summer': "2015-06-21",
        'solstice_winter': "2015-12-21",
    }


class TestWithFixtures:
    """Tests using pytest fixtures."""

    def test_multiple_locations_summer_solstice(self, sample_locations, sample_dates):
        """Test multiple locations on summer solstice."""
        date = sample_dates['solstice_summer']

        for location_name, (lat, lon) in sample_locations.items():
            result = get_ozone_conc(lat, lon, date)
            assert isinstance(result, np.ndarray)
            assert result[0] > 0
            assert result[0] < 500  # Reasonable upper bound

    def test_single_location_all_seasons(self, sample_locations, sample_dates):
        """Test single location across different seasons."""
        lat, lon = sample_locations['new_york']

        seasonal_dates = [
            sample_dates['winter_nh'],
            sample_dates['spring_nh'],
            sample_dates['summer_nh'],
            sample_dates['fall_nh'],
        ]

        results = []
        for date in seasonal_dates:
            result = get_ozone_conc(lat, lon, date)
            results.append(result[0])

        # All results should be positive and reasonable
        assert all(r > 0 for r in results)
        assert all(r < 500 for r in results)

        # Results should vary across seasons
        assert len(set(results)) == 4  # All different values
