import numpy as np
from numpy.testing import assert_allclose
from math import pi

from ..coordinate_transforms import sph2cart


def test_sph2cart() -> None:
    # Test case 1: Zero radius
    rad = 0
    lat = pi / 4  # 45 degrees
    lon = pi / 2  # 90 degrees
    expected_result = np.array([0, 0, 0])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result)

    # Test case 2: Non-zero radius and lat/lon
    rad = 1
    lat = pi / 3  # 60 degrees
    lon = pi / 4  # 45 degrees
    expected_result = np.array([0.353553, 0.353553, 0.866025])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 3: Negative radius and lat/lon (should raise ValueError)
    rad = -1
    lat = -pi / 6  # -30 degrees
    lon = -pi / 3  # -60 degrees
    try:
        result = sph2cart(rad, lat, lon)
        assert False  # The line above should raise an exception, so this line should not be reached
    except ValueError:
        assert True

    # Test case 4: Large radius and lat/lon
    rad = 1000
    lat = 0
    lon = pi
    expected_result = np.array([-1000, 0, 0])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 5: Radius and lat/lon near boundaries
    rad = 1e-6
    lat = -pi / 2 + 1e-9  # Just below the south pole
    lon = pi  # 180 degrees
    expected_result = np.array([0, 0, -1e-6])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 6: Radius and lat/lon at the north pole
    rad = 2
    lat = pi / 2  # 90 degrees
    lon = 0  # 0 degrees
    expected_result = np.array([0, 0, 2])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Test case 7: Radius and lat/lon at the prime meridian
    rad = 3
    lat = 0  # 0 degrees
    lon = 0  # 0 degrees
    expected_result = np.array([3, 0, 0])
    result = sph2cart(rad, lat, lon)
    assert_allclose(result, expected_result, atol=1e-6)

    # Additional test cases...
    # TODO: Add more test cases as needed
