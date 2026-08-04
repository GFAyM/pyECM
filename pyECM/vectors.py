"""Vector utilities for pyECM."""

import numpy as np


def remove_zeros(x, y, z=None):
    """Remove zeros values from input arrays.

    :param x: First array to be treated
    :type x: array
    :param y: Second array to be treated
    :type y: array
    :param z: Third array to be treated, defaults to None
    :type z: array, optional
    :return: x, y, z without theirs zero values
    :rtype: arrays
    """
    index = 0
    indices_to_delete = np.array([])

    if z is None:
        for i in x:
            if (i == 0) and (y[index] == 0):
                indices_to_delete = np.append(indices_to_delete, index)
            index = index + 1
        indices_to_delete = indices_to_delete.astype(int)
        result = (
            np.delete(x, indices_to_delete),
            np.delete(y, indices_to_delete),
            indices_to_delete,
        )
    else:
        for i in x:
            if (i == 0) and (y[index] == 0) and (z[index] == 0):
                indices_to_delete = np.append(indices_to_delete, index)
            index = index + 1
        indices_to_delete = indices_to_delete.astype(int)
        result = (
            np.delete(x, indices_to_delete),
            np.delete(y, indices_to_delete),
            np.delete(z, indices_to_delete),
            indices_to_delete,
        )

    return result


def normalize_vector(x, y, z=None):
    """Normalize 2D/3D vector (to unity).

    :param x: x component/s
    :type x: float or array
    :param y: y component/s
    :type y: float or array
    :param z: z component/s, defaults to None
    :type z: float or array, optional
    :return: Normalized vector/s
    :rtype: float or array
    """
    # z = np.array(z)
    if np.array(z).all() is None:
        n = np.sqrt(np.power(x, 2) + np.power(y, 2))
        versor = [x / n, y / n]
    else:
        n = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
        versor = [x / n, y / n, z / n]
    return versor
