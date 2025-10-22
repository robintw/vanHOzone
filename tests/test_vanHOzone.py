"""
Comprehensive tests for the vanHOzone package.

Tests the van Heuklon (1979) Ozone model implementation for estimating
atmospheric ozone concentration.
"""
import unittest
import numpy as np
from datetime import datetime
from vanHOzone import get_ozone_conc


class TestGetOzoneConc(unittest.TestCase):
    """Test cases for the get_ozone_conc function."""

    def assertArrayAlmostEqual(self, arr1, arr2, places=7):
        """Helper to assert numpy arrays are almost equal."""
        np.testing.assert_array_almost_equal(arr1, arr2, decimal=places)

    def test_scalar_northern_hemisphere(self):
        """Test scalar values in Northern hemisphere (London)."""
        lat = 51.5
        lon = -0.1
        timestamp = "2015-06-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([383.85882587])
        self.assertArrayAlmostEqual(result, expected)

    def test_scalar_southern_hemisphere(self):
        """Test scalar values in Southern hemisphere (Sydney)."""
        lat = -33.9
        lon = 151.2
        timestamp = "2015-12-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([313.50199155])
        self.assertArrayAlmostEqual(result, expected)

    def test_scalar_equator(self):
        """Test scalar values at the Equator (Quito)."""
        lat = 0.0
        lon = -78.5
        timestamp = "2015-03-21"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([235.0])
        self.assertArrayAlmostEqual(result, expected)

    def test_datetime_object(self):
        """Test using a datetime object instead of string."""
        lat = 40.7
        lon = -74.0
        dt = datetime(2015, 9, 23, 12, 0, 0)
        result = get_ozone_conc(lat, lon, dt)
        expected = np.array([316.87971504])
        self.assertArrayAlmostEqual(result, expected)

    def test_array_inputs_same_timestamp(self):
        """Test array inputs with same timestamp for all locations."""
        lats = [51.5, -33.9, 0.0, 40.7, -23.5]
        lons = [-0.1, 151.2, -78.5, -74.0, -46.6]
        timestamp = "2015-06-21"
        result = get_ozone_conc(lats, lons, timestamp)
        expected = np.array([381.33586549, 289.17339348, 235.0,
                            352.69969946, 267.86515712])
        self.assertArrayAlmostEqual(result, expected)

    def test_array_inputs_array_timestamps(self):
        """Test array inputs with array of timestamps."""
        lats = [51.5, -33.9, 0.0]
        lons = [-0.1, 151.2, -78.5]
        timestamps = ["2015-01-15", "2015-07-15", "2015-12-15"]
        result = get_ozone_conc(lats, lons, timestamps)
        expected = np.array([351.4262453, 295.71418863, 235.0])
        self.assertArrayAlmostEqual(result, expected)

    def test_northern_hemisphere_winter(self):
        """Test Northern hemisphere in winter (New York)."""
        lat = 40.7
        lon = -74.0
        timestamp = "2015-01-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([330.35930121])
        self.assertArrayAlmostEqual(result, expected)

    def test_northern_hemisphere_spring(self):
        """Test Northern hemisphere in spring (New York)."""
        lat = 40.7
        lon = -74.0
        timestamp = "2015-04-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([360.65695415])
        self.assertArrayAlmostEqual(result, expected)

    def test_northern_hemisphere_summer(self):
        """Test Northern hemisphere in summer (New York)."""
        lat = 40.7
        lon = -74.0
        timestamp = "2015-07-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([343.68467842])
        self.assertArrayAlmostEqual(result, expected)

    def test_northern_hemisphere_fall(self):
        """Test Northern hemisphere in fall (New York)."""
        lat = 40.7
        lon = -74.0
        timestamp = "2015-10-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([312.72004167])
        self.assertArrayAlmostEqual(result, expected)

    def test_southern_hemisphere_summer(self):
        """Test Southern hemisphere in summer (Melbourne)."""
        lat = -37.8
        lon = 144.9
        timestamp = "2015-01-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([319.17149283])
        self.assertArrayAlmostEqual(result, expected)

    def test_southern_hemisphere_fall(self):
        """Test Southern hemisphere in fall (Melbourne)."""
        lat = -37.8
        lon = 144.9
        timestamp = "2015-04-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([293.71526119])
        self.assertArrayAlmostEqual(result, expected)

    def test_southern_hemisphere_winter(self):
        """Test Southern hemisphere in winter (Melbourne)."""
        lat = -37.8
        lon = 144.9
        timestamp = "2015-07-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([308.07017292])
        self.assertArrayAlmostEqual(result, expected)

    def test_southern_hemisphere_spring(self):
        """Test Southern hemisphere in spring (Melbourne)."""
        lat = -37.8
        lon = 144.9
        timestamp = "2015-10-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([334.08756962])
        self.assertArrayAlmostEqual(result, expected)

    def test_extreme_latitude_arctic(self):
        """Test extreme latitude in Arctic region."""
        lat = 80
        lon = 0.0
        timestamp = "2015-06-21"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([402.56685809])
        self.assertArrayAlmostEqual(result, expected)

    def test_extreme_latitude_antarctic(self):
        """Test extreme latitude in Antarctic region."""
        lat = -80
        lon = 0.0
        timestamp = "2015-06-21"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([288.11037702])
        self.assertArrayAlmostEqual(result, expected)

    def test_different_longitudes_west(self):
        """Test Northern hemisphere with western longitude."""
        lat = 45.0
        lon = -120
        timestamp = "2015-06-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([362.38905981])
        self.assertArrayAlmostEqual(result, expected)

    def test_different_longitudes_prime_meridian(self):
        """Test Northern hemisphere at Prime Meridian."""
        lat = 45.0
        lon = 0
        timestamp = "2015-06-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([362.38905981])
        self.assertArrayAlmostEqual(result, expected)

    def test_different_longitudes_east(self):
        """Test Northern hemisphere with eastern longitude."""
        lat = 45.0
        lon = 120
        timestamp = "2015-06-15"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([374.73667068])
        self.assertArrayAlmostEqual(result, expected)

    def test_iso8601_datetime_format(self):
        """Test ISO 8601 datetime format with time component."""
        lat = 35.0
        lon = 139.7
        timestamp = "2015-08-30T14:30:00"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([308.48281735])
        self.assertArrayAlmostEqual(result, expected)

    def test_date_format_iso(self):
        """Test ISO date format (YYYY-MM-DD)."""
        lat = 52.5
        lon = 13.4
        timestamp = "2015-05-20"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([411.43655659])
        self.assertArrayAlmostEqual(result, expected)

    def test_date_format_us(self):
        """Test US date format (MM/DD/YYYY)."""
        lat = 52.5
        lon = 13.4
        timestamp = "05/20/2015"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([411.43655659])
        self.assertArrayAlmostEqual(result, expected)

    def test_date_format_european(self):
        """Test European date format (DD/MM/YYYY)."""
        lat = 52.5
        lon = 13.4
        timestamp = "20/05/2015"
        result = get_ozone_conc(lat, lon, timestamp)
        expected = np.array([411.43655659])
        self.assertArrayAlmostEqual(result, expected)

    def test_numpy_arrays_as_input(self):
        """Test that numpy arrays work as inputs."""
        lats = np.array([51.5, -33.9, 0.0])
        lons = np.array([-0.1, 151.2, -78.5])
        timestamp = "2015-06-21"
        result = get_ozone_conc(lats, lons, timestamp)
        expected = np.array([381.33586549, 289.17339348, 235.0])
        self.assertArrayAlmostEqual(result, expected)

    def test_datetime_array(self):
        """Test with array of datetime objects."""
        lats = [51.5, -33.9, 0.0]
        lons = [-0.1, 151.2, -78.5]
        timestamps = [datetime(2015, 1, 15), datetime(2015, 7, 15),
                     datetime(2015, 12, 15)]
        result = get_ozone_conc(lats, lons, timestamps)
        expected = np.array([351.4262453, 295.71418863, 235.0])
        self.assertArrayAlmostEqual(result, expected)

    def test_mismatched_lat_lon_lengths(self):
        """Test that mismatched lat and lon arrays raise ValueError."""
        lats = [51.5, -33.9, 0.0]
        lons = [-0.1, 151.2]  # Different length
        timestamp = "2015-06-21"
        with self.assertRaises(ValueError) as context:
            get_ozone_conc(lats, lons, timestamp)
        self.assertIn("lan and lon", str(context.exception))

    def test_mismatched_timestamp_length(self):
        """Test that mismatched timestamp array length raises an error."""
        lats = [51.5, -33.9, 0.0]
        lons = [-0.1, 151.2, -78.5]
        timestamps = ["2015-01-15", "2015-07-15"]  # Different length
        # Due to the exception handling in the code, this raises TypeError
        with self.assertRaises(TypeError):
            get_ozone_conc(lats, lons, timestamps)

    def test_result_is_numpy_array(self):
        """Test that result is always a numpy array."""
        lat = 51.5
        lon = -0.1
        timestamp = "2015-06-15"
        result = get_ozone_conc(lat, lon, timestamp)
        self.assertIsInstance(result, np.ndarray)

    def test_result_shape_scalar(self):
        """Test that scalar inputs return array of shape (1,)."""
        lat = 51.5
        lon = -0.1
        timestamp = "2015-06-15"
        result = get_ozone_conc(lat, lon, timestamp)
        self.assertEqual(result.shape, (1,))

    def test_result_shape_array(self):
        """Test that array inputs return array of correct shape."""
        lats = [51.5, -33.9, 0.0, 40.7, -23.5]
        lons = [-0.1, 151.2, -78.5, -74.0, -46.6]
        timestamp = "2015-06-21"
        result = get_ozone_conc(lats, lons, timestamp)
        self.assertEqual(result.shape, (5,))

    def test_positive_ozone_values(self):
        """Test that ozone concentration values are always positive."""
        lats = [80, 45, 0, -45, -80]
        lons = [0, 90, -90, 180, -180]
        timestamp = "2015-06-21"
        result = get_ozone_conc(lats, lons, timestamp)
        self.assertTrue(np.all(result > 0))

    def test_leap_year_date(self):
        """Test with a leap year date."""
        lat = 51.5
        lon = -0.1
        timestamp = "2016-02-29"  # Leap year
        result = get_ozone_conc(lat, lon, timestamp)
        # Just check it doesn't crash and returns a valid result
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result[0] > 0)

    def test_year_boundaries(self):
        """Test dates at year boundaries."""
        lat = 40.7
        lon = -74.0

        # New Year's Day
        result_jan1 = get_ozone_conc(lat, lon, "2015-01-01")
        self.assertTrue(result_jan1[0] > 0)

        # New Year's Eve
        result_dec31 = get_ozone_conc(lat, lon, "2015-12-31")
        self.assertTrue(result_dec31[0] > 0)


if __name__ == '__main__':
    unittest.main()
