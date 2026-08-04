"""Preloaded molecule."""

import numpy as np


def mol_SeIClO_rotated(enantiomero, campovectorial=None):
    """Preloaded SeIClO molecule.

    :param enantiomero: R- or S-enantiomer
    :type enantiomero: string
    :param campovectorial: vectorial field, defaults to None
    :type campovectorial: deprecated, optional
    :return: nro_atomos, tuple, NDArray, NDArray
    :rtype: integer,
    """
    # Angstrom Units
    nro_atomos = 4
    coordenadas_x = np.zeros(nro_atomos)
    coordenadas_y = np.zeros(nro_atomos)
    coordenadas_z = np.zeros(nro_atomos)
    coordenadas_color = [None] * nro_atomos
    nombres = [None] * nro_atomos
    bonds = np.array([[0, 1], [0, 2], [0, 3]])
    if enantiomero == "R":
        (
            coordenadas_x[0],
            coordenadas_y[0],
            coordenadas_z[0],
            coordenadas_color[0],
            nombres[0],
        ) = (0.0, 0.0, 0.0, "orange", "Se")
        (
            coordenadas_x[1],
            coordenadas_y[1],
            coordenadas_z[1],
            coordenadas_color[1],
            nombres[1],
        ) = (-0.803, -0.353, -1.347, "red", "O")
        (
            coordenadas_x[2],
            coordenadas_y[2],
            coordenadas_z[2],
            coordenadas_color[2],
            nombres[2],
        ) = (-0.567, 2.513, 0.599, "purple", "I")
        (
            coordenadas_x[3],
            coordenadas_y[3],
            coordenadas_z[3],
            coordenadas_color[3],
            nombres[3],
        ) = (2.152, 0.238, -0.595, "green", "Cl")
        dipolo = np.array([0.000, 0.000, 2.251])  # Debyes
    elif enantiomero == "S":
        (
            coordenadas_x[0],
            coordenadas_y[0],
            coordenadas_z[0],
            coordenadas_color[0],
            nombres[0],
        ) = (0.0, 0.0, 0.0, "orange", "Se")
        (
            coordenadas_x[1],
            coordenadas_y[1],
            coordenadas_z[1],
            coordenadas_color[1],
            nombres[1],
        ) = (-0.803, -0.353, 1.347, "red", "O")
        (
            coordenadas_x[2],
            coordenadas_y[2],
            coordenadas_z[2],
            coordenadas_color[2],
            nombres[2],
        ) = (-0.567, 2.513, -0.599, "purple", "I")
        (
            coordenadas_x[3],
            coordenadas_y[3],
            coordenadas_z[3],
            coordenadas_color[3],
            nombres[3],
        ) = (2.152, 0.238, 0.595, "green", "Cl")
        dipolo = np.array([0.000, 0.000, -2.251])  # Debyes
    atomos = (coordenadas_x, coordenadas_y, coordenadas_z, coordenadas_color, nombres)
    return nro_atomos, atomos, dipolo, bonds
