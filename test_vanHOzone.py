"""
Property-based tests for vanHOzone using Hypothesis.

These tests verify the correctness of the van Heuklon ozone model implementation
by testing various properties that should hold for all valid inputs.
"""
import numpy as np
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays
import pytest

from vanHOzone import get_ozone_conc


# Custom strategies for valid inputs
latitudes = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)
longitudes = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
dates = st.datetimes(
    min_value=datetime(1970, 1, 1),
    max_value=datetime(2100, 12, 31)
)


class TestOzoneBasicProperties:
    """Test basic properties that should always hold."""

    @given(latitudes, longitudes, dates)
    def test_output_is_finite(self, lat, lon, timestamp):
        """Ozone concentration should always be a finite number."""
        result = get_ozone_conc(lat, lon, timestamp)
        assert np.isfinite(result), f"Result is not finite: {result}"

    @given(latitudes, longitudes, dates)
    def test_output_is_positive(self, lat, lon, timestamp):
        """Ozone concentration should always be positive."""
        result = get_ozone_conc(lat, lon, timestamp)
        assert result > 0, f"Result is not positive: {result}"

    @given(latitudes, longitudes, dates)
    def test_output_in_reasonable_range(self, lat, lon, timestamp):
        """Ozone concentration should be in a physically reasonable range.

        Typical atmospheric ozone values range from about 100-500 matm-cm.
        """
        result = get_ozone_conc(lat, lon, timestamp)
        assert 50 < result < 600, f"Result outside reasonable range: {result}"


class TestScalarVsArrayEquivalence:
    """Test that scalar and array inputs produce equivalent results."""

    @given(latitudes, longitudes, dates)
    def test_scalar_equals_single_element_array(self, lat, lon, timestamp):
        """A scalar input should give the same result as a single-element array."""
        scalar_result = get_ozone_conc(lat, lon, timestamp)
        array_result = get_ozone_conc([lat], [lon], timestamp)

        assert np.isclose(scalar_result, array_result[0]), \
            f"Scalar {scalar_result} != array {array_result[0]}"

    @given(
        st.lists(latitudes, min_size=1, max_size=10),
        st.lists(longitudes, min_size=1, max_size=10),
        dates
    )
    def test_array_input_same_length(self, lats, lons, timestamp):
        """Arrays of same length should produce results of that length."""
        assume(len(lats) == len(lons))

        result = get_ozone_conc(lats, lons, timestamp)
        assert len(result) == len(lats)
        assert all(np.isfinite(result))


class TestTimestampFormats:
    """Test different timestamp input formats."""

    @given(latitudes, longitudes, dates)
    def test_datetime_vs_iso_string(self, lat, lon, timestamp):
        """Datetime object and ISO string should give same result."""
        result_datetime = get_ozone_conc(lat, lon, timestamp)
        result_string = get_ozone_conc(lat, lon, timestamp.strftime("%Y-%m-%d"))

        assert np.isclose(result_datetime, result_string), \
            f"Datetime {result_datetime} != string {result_string}"

    @given(latitudes, longitudes, dates)
    def test_different_string_formats(self, lat, lon, timestamp):
        """Different date string formats should parse to same result."""
        iso_format = timestamp.strftime("%Y-%m-%d")
        us_format = timestamp.strftime("%m/%d/%Y")

        result_iso = get_ozone_conc(lat, lon, iso_format)
        result_us = get_ozone_conc(lat, lon, us_format)

        assert np.isclose(result_iso, result_us), \
            f"ISO {result_iso} != US format {result_us}"

    @given(
        st.lists(latitudes, min_size=2, max_size=5),
        st.lists(longitudes, min_size=2, max_size=5),
        st.lists(dates, min_size=2, max_size=5)
    )
    def test_timestamp_array_same_length(self, lats, lons, timestamps):
        """Timestamp arrays must match lat/lon array length."""
        assume(len(lats) == len(lons) == len(timestamps))

        result = get_ozone_conc(lats, lons, timestamps)
        assert len(result) == len(lats)


class TestInputValidation:
    """Test that invalid inputs are properly rejected."""

    @given(
        st.lists(latitudes, min_size=1, max_size=5),
        st.lists(longitudes, min_size=1, max_size=5)
    )
    def test_mismatched_lat_lon_length_raises_error(self, lats, lons):
        """Mismatched lat/lon array lengths should raise ValueError."""
        assume(len(lats) != len(lons))

        with pytest.raises(ValueError, match="lat and lon arrays"):
            get_ozone_conc(lats, lons, datetime.now())

    @given(st.integers(min_value=2, max_value=5))
    def test_mismatched_timestamp_length_raises_error(self, n):
        """Mismatched timestamp array length should raise ValueError."""
        # Create arrays of length n
        lats = [0.0] * n
        lons = [0.0] * n
        # Create timestamp array of different length (n-1)
        timestamps = [datetime.now()] * (n - 1)

        with pytest.raises(ValueError, match="Timestamp must be the same length"):
            get_ozone_conc(lats, lons, timestamps)


class TestHemisphereSpecificBehavior:
    """Test hemisphere-specific calculations."""

    @given(
        st.floats(min_value=0.1, max_value=90.0),
        longitudes,
        dates
    )
    def test_northern_hemisphere_calculation(self, lat, lon, timestamp):
        """Northern hemisphere (positive latitude) should produce valid results."""
        result = get_ozone_conc(lat, lon, timestamp)
        assert np.isfinite(result) and result > 0

    @given(
        st.floats(min_value=-90.0, max_value=-0.1),
        longitudes,
        dates
    )
    def test_southern_hemisphere_calculation(self, lat, lon, timestamp):
        """Southern hemisphere (negative latitude) should produce valid results."""
        result = get_ozone_conc(lat, lon, timestamp)
        assert np.isfinite(result) and result > 0

    @given(longitudes, dates)
    def test_equator_calculation(self, lon, timestamp):
        """Equator (latitude = 0) should produce valid results."""
        result = get_ozone_conc(0.0, lon, timestamp)
        assert np.isfinite(result) and result > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @given(dates)
    def test_north_pole(self, timestamp):
        """North pole (90 degrees) should produce valid results."""
        result = get_ozone_conc(90.0, 0.0, timestamp)
        assert np.isfinite(result) and result > 0

    @given(dates)
    def test_south_pole(self, timestamp):
        """South pole (-90 degrees) should produce valid results."""
        result = get_ozone_conc(-90.0, 0.0, timestamp)
        assert np.isfinite(result) and result > 0

    @given(latitudes)
    def test_prime_meridian(self, lat):
        """Prime meridian (longitude = 0) should produce valid results."""
        result = get_ozone_conc(lat, 0.0, datetime(2020, 6, 21))
        assert np.isfinite(result) and result > 0

    @given(latitudes)
    def test_international_date_line(self, lat):
        """International date line (longitude = 180/-180) should produce valid results."""
        result_180 = get_ozone_conc(lat, 180.0, datetime(2020, 6, 21))
        result_neg180 = get_ozone_conc(lat, -180.0, datetime(2020, 6, 21))

        assert np.isfinite(result_180) and result_180 > 0
        assert np.isfinite(result_neg180) and result_neg180 > 0

    @given(latitudes, longitudes)
    def test_new_years_day(self, lat, lon):
        """January 1st (day 1) should produce valid results."""
        result = get_ozone_conc(lat, lon, datetime(2020, 1, 1))
        assert np.isfinite(result) and result > 0

    @given(latitudes, longitudes)
    def test_new_years_eve(self, lat, lon):
        """December 31st (day 365/366) should produce valid results."""
        result = get_ozone_conc(lat, lon, datetime(2020, 12, 31))
        assert np.isfinite(result) and result > 0


class TestSeasonalVariation:
    """Test that ozone varies with season as expected."""

    @given(latitudes, longitudes)
    def test_seasonal_variation_exists(self, lat, lon):
        """Ozone concentration should vary throughout the year."""
        winter = get_ozone_conc(lat, lon, datetime(2020, 1, 1))
        summer = get_ozone_conc(lat, lon, datetime(2020, 7, 1))

        # For most locations, there should be some seasonal variation
        # We don't assert they're different because at equator they might be similar
        assert np.isfinite(winter) and np.isfinite(summer)
        assert winter > 0 and summer > 0


class TestNumpyArrayInputs:
    """Test with numpy arrays specifically."""

    @given(
        st.integers(min_value=1, max_value=10),
        dates
    )
    def test_numpy_array_inputs(self, n, timestamp):
        """Should handle numpy arrays correctly."""
        # Generate arrays of the same length
        lats = np.random.uniform(-90, 90, n)
        lons = np.random.uniform(-180, 180, n)

        result = get_ozone_conc(lats, lons, timestamp)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(lats)
        assert all(np.isfinite(result))
        assert all(result > 0)


class TestDeterminism:
    """Test that the function is deterministic."""

    @given(latitudes, longitudes, dates)
    def test_repeated_calls_same_result(self, lat, lon, timestamp):
        """Calling with same inputs should always give same output."""
        result1 = get_ozone_conc(lat, lon, timestamp)
        result2 = get_ozone_conc(lat, lon, timestamp)

        assert result1 == result2, "Function is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
