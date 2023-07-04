import numpy as np


def sph2cart(rad: float, lat: float, lon: float) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Args:
        rad (float): The radius (must be non-negative).
        lat (float): The latitude in radians.
        lon (float): The longitude in radians.

    Returns:
        np.ndarray: An array containing the Cartesian coordinates [x, y, z].
    """
    if rad < 0:
        raise ValueError("Radius must be non-negative.")

    cos_lat = np.cos(lat)
    x = rad * cos_lat * np.cos(lon)
    y = rad * cos_lat * np.sin(lon)
    z = rad * np.sin(lat)

    return np.asarray([x, y, z])



def cart2sph(x: float, y: float, z: float) -> np.ndarray:
    rad = np.sqrt(x*x + y*y + z*z)
    lat = np.arcsin(z / rad)
    lon = np.arctan2(y, x)
    
    return np.asarray([rad, lat, lon])


def geo2cyl(point: list, rotation_matrix: np.ndarray) -> list:
    """
    Convert geographical coordinates to cylindrical coordinates in the
    seismic frame.

    Args:
        point (list): [radial position in m, latitude in rad, longitude in rad]

    Returns:
        list: [radial position in m, vertical position in m, azimuth from
        source in rad]
    """
    radial_pos = point[0]
    latitude = point[1]
    longitude = point[2]

    cos_lat = np.cos(latitude)
    sin_lat = np.sin(latitude)
    cos_lon = np.cos(longitude)
    sin_lon = np.sin(longitude)

    # Transform geographical coordinates to Cartesian coordinates in the
    # Earth frame
    x_earth = radial_pos * cos_lat * cos_lon
    y_earth = radial_pos * cos_lat * sin_lon
    z_earth = radial_pos * sin_lat

    # Rotate coordinates from the Earth frame to the source frame
    rotated_coords = np.matmul(rotation_matrix.transpose(), np.asarray([x_earth, y_earth, z_earth]))
    x, y, z = rotated_coords

    # Convert to cylindrical coordinates in the source frame
    s = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return [s, z, phi]


def cart2polar(self, s: float, z: float) -> list:
    """Transforms inplane cylindrical coords (cartesian)
    to polar coords

    Args:
        s (float): distance from cylindarical axis in m
        z (float): distance along cylindrical axis in m

    Returns:
        list: [radius, theta]
    """       
    r = np.sqrt(s**2 + z**2)
    theta = np.arctan(z/s)
    
    return [r, theta]