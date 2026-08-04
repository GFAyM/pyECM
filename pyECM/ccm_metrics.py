"""Continuous Chirality Measure for pyECM.

The ccm function does not depend on the molecule instance.
"""

import mendeleev
import numpy as np


def ccm(positions, atom_names, positions_achiral=None, z_coordinate=1.00):
    """Obtain CCM (two definitions are available for it) for a molecule.

    :param positions: array (n_atoms, 3) for the structure
    :param atom_names: atom names
    :param positions_achiral: array (n_atoms, 3) for the reference achiral structure. If
        None, z=0.
    :param z_coordinate: scale factor over z coordinate
    :return: Norm_1, CCM_1, Norm_2, CCM_2
    """
    """Obtain the CCM (Continuous Chirality Measure) for a molecule, using
    two different normalization definitions.

    :param positions: nuclear positions of the (chiral) structure
    :type positions: numpy.ndarray, shape (n_atoms, 3)
    :param atom_names: atom names, used to look up atomic weights
        (e.g. "H", "O") for the center-of-mass calculation
    :type atom_names: sequence of str
    :param positions_achiral: nuclear positions of the reference achiral
        structure. If None, the achiral reference is derived from
        `positions` by zeroing its z coordinate, defaults to None
    :type positions_achiral: numpy.ndarray, shape (n_atoms, 3), optional
    :param z_coordinate: scale factor applied to the z coordinate of
        `positions` before computing the distance to the achiral
        reference, defaults to 1.00
    :type z_coordinate: float, optional
    :return: Norm_1, CCM_1 (1998), Norm_2, CCM_2 (2010)
    :rtype: tuple(float, float, float, float)
    """

    n_atoms = len(positions)
    x_coordinates = positions[:, 0]
    y_coordinates = positions[:, 1]
    coordenadas_z = positions[:, 2] * z_coordinate

    if positions_achiral is None:
        achiral_x_coordinates = x_coordinates
        achiral_y_coordinates = y_coordinates
        achiral_z_coordinates = coordenadas_z * 0
    else:
        achiral_x_coordinates = positions_achiral[:, 0]
        achiral_y_coordinates = positions_achiral[:, 1]
        achiral_z_coordinates = positions_achiral[:, 2]

    diff_x = x_coordinates - achiral_x_coordinates
    diff_y = y_coordinates - achiral_y_coordinates
    diff_z = coordenadas_z - achiral_z_coordinates
    CCM_distance = np.sum(diff_x**2 + diff_y**2 + diff_z**2)

    mean_x = np.average(x_coordinates)
    mean_y = np.average(y_coordinates)
    mean_z = np.average(coordenadas_z)

    # Method 1
    # https://doi.org/10.1021/ja9800941 (Eq. 1)
    atomic_weights = np.array(
        [getattr(mendeleev, name).atomic_weight for name in atom_names]
    )

    total_mass = np.sum(atomic_weights)
    center_of_mass_x = np.sum(x_coordinates * atomic_weights) / total_mass
    center_of_mass_y = np.sum(y_coordinates * atomic_weights) / total_mass
    center_of_mass_z = np.sum(coordenadas_z * atomic_weights) / total_mass
    distances_to_CM_x = x_coordinates - center_of_mass_x
    distances_to_CM_y = y_coordinates - center_of_mass_y
    distances_to_CM_z = coordenadas_z - center_of_mass_z
    distances_to_CM = np.sqrt(
        distances_to_CM_x**2 + distances_to_CM_y**2 + distances_to_CM_z**2
    )

    D = np.max(distances_to_CM)
    Norm_1 = D**2 * n_atoms
    CCM_1 = (1 / Norm_1) * CCM_distance * 100

    # Method 2
    # https://doi.org/10.1002/chir.20807 (Eq. 1)
    Norm_2 = np.sum(
        (x_coordinates - mean_x) ** 2
        + (y_coordinates - mean_y) ** 2
        + (coordenadas_z - mean_z) ** 2
    )
    CCM_2 = CCM_distance / Norm_2 * 100

    return Norm_1, CCM_1, Norm_2, CCM_2
