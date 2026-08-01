"""XYZ file I/O utilities for pyECM.molecula objects.
"""
import os

import numpy as np


def read_xyz(filename):
    """Read xyz file.

    :param filename: xyz file name
    :return: (n_atoms, coord_x, coord_y, coord_z, colors, names)
    """
    # Check for empty lines in xyz file.
    with open(filename) as xyz_check:
        n_atoms_check = int(xyz_check.readline())
        xyz_check.readline()  # title
        j = 0
        for line in xyz_check:
            j += 1
            if j > n_atoms_check:
                raise TypeError("Error in xyz file format. Check for empty lines.")

    with open(filename) as xyz:
        n_atoms = int(xyz.readline())
        xyz.readline()  # title

        coord_x = np.zeros(n_atoms)
        coord_y = np.zeros(n_atoms)
        coord_z = np.zeros(n_atoms)
        colors = ["black"] * n_atoms
        names = [None] * n_atoms

        i = 0
        for line in xyz:
            atom, x, y, z = line.split()
            coord_x[i] = x
            coord_y[i] = y
            coord_z[i] = z
            names[i] = atom
            i += 1

    return n_atoms, coord_x, coord_y, coord_z, colors, names


def atom_symbol(raw_name):
    """Remove numbers from atom names (e.g. 'H1' -> 'H')."""
    return "".join([c for c in str(raw_name) if not c.isdigit()])


def write_xyz(filename, nro_atoms, atom_names, positions, z_coordinate=1.0):
    """Writes xyz files from corresponding atomic names and positions."""
    with open(filename, "w") as f:
        f.write(str(nro_atoms) + "\n")
        f.write("XYZ file created by pyECM\n")
        for j in range(nro_atoms):
            f.write(
                atom_names[j]
                + "   "
                + "{:.6f}".format(positions[j][0])
                + " "
                + "{:.6f}".format(positions[j][1])
                + " "
                + "{:.6f}".format(positions[j][2] * z_coordinate)
                + "\n"
            )