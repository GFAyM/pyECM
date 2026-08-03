"""Geometry utilities for pyECM.molecula objects.

Pure functions operating on plain arrays of atom data.
"""

import numpy as np


def build_positions(n_atoms, atoms):
    """Build a (n_atoms, 3) array of positions from an 'atoms' tuple
    (coord_x, coord_y, coord_z, colors, names), as returned by
    xyz_io.read_xyz.

    :param n_atoms: number of atoms
    :type n_atoms: int
    :param atoms: (coord_x, coord_y, coord_z, colors, names) tuple
    :type atoms: tuple
    :return: positions array
    :rtype: numpy.ndarray, shape (n_atoms, 3)
    """
    positions = []
    for i in range(n_atoms):
        positions.append([atoms[0][i], atoms[1][i], atoms[2][i]])
    return np.array(positions)


def central_point_from_origin(n_atoms, positions, atom_names, origin):
    """Resolve the central point from 'origin': either the position of
    the named atom (if origin is a string) or the given point directly
    (if origin is a numpy array).

    :param n_atoms: number of atoms
    :type n_atoms: int
    :param positions: (n_atoms, 3) array of positions
    :type positions: numpy.ndarray
    :param atom_names: atom name/label per atom, used to find 'origin'
        when it's given as a string
    :type atom_names: sequence of str
    :param origin: point (numpy.ndarray) or atom name (str) used as origin
    :type origin: numpy.ndarray or str or None
    :return: the resolved central point, or None if origin is None or the
        named atom wasn't found
    :rtype: numpy.ndarray or None
    """
    if origin is None:
        return None
    if isinstance(origin, str):
        for i in range(n_atoms):
            if atom_names[i] == origin:
                return np.array([positions[i][0], positions[i][1], positions[i][2]])
        return None
    if isinstance(origin, np.ndarray):
        return origin
    return None


def shift_positions_to_origin(n_atoms, positions, atom_names, origin):
    """Shift 'positions' so that 'origin' becomes the new coordinate
    origin.

    :param n_atoms: number of atoms
    :type n_atoms: int
    :param positions: (n_atoms, 3) array of positions
    :type positions: numpy.ndarray
    :param atom_names: atom name/label per atom, used to find 'origin'
        when it's given as a string
    :type atom_names: sequence of str
    :param origin: point (numpy.ndarray) or atom name (str) used as origin
    :type origin: numpy.ndarray or str
    :return: (new_positions, central_point_coordinates)
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    central_point_coordinates = None
    if isinstance(origin, str):
        for i in range(n_atoms):
            if atom_names[i] == origin:
                central_point_coordinates = np.array(
                    [positions[i][0], positions[i][1], positions[i][2]]
                )
    if isinstance(origin, np.ndarray):
        central_point_coordinates = origin

    new_positions = np.zeros(3)
    for i in range(n_atoms):
        new_line = positions[i] - central_point_coordinates
        new_positions = np.vstack([new_positions, new_line])
    new_positions = np.delete(new_positions, (0), axis=0)  # drop the zeros row

    return new_positions, central_point_coordinates
