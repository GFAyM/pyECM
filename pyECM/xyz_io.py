"""XYZ file I/O utilities for pyECM.molecula objects.
The functions are pure, in the sense that they receive and return
data without reading nor writing attributes from a molecule instance.
"""

import numpy as np


def read_xyz(filename):
    """Read a xyz file into plain arrays.

    :param filename: path to the xyz file
    :type filename: str
    :raises TypeError: if the file has empty lines between the header
        and the last atom line (a common formatting error that silently
        breaks column-based readers)
    :return: (n_atoms, coord_x, coord_y, coord_z, colors, names).
        `colors` is initialized to "black" for every atom
        (colors are not stored in the xyz format itself;
        they are assigned elsewhere)
    :rtype: tuple(int, numpy.ndarray, numpy.ndarray, numpy.ndarray,
        list of str, list of str)
    """
    # Check for empty lines in xyz file.
    with open(filename) as xyz_check:
        n_atoms_check = int(xyz_check.readline())
        xyz_check.readline()  # title
        j = 0
        for _line in xyz_check:
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
    """Remove trailing digits from an atom label (e.g. 'H1' -> 'H'),
    typically used to turn a per-atom identifier (unique within a
    molecule) back into its chemical element symbol.

    :param raw_name: atom label, possibly with a trailing index number
    :type raw_name: str or any type convertible to str
    :return: the label with all digit characters removed
    :rtype: str
    """
    return "".join([c for c in str(raw_name) if not c.isdigit()])


def write_xyz(filename, n_atoms, atom_names, positions, z_coordinate=1.0):
    """Write a xyz file from atom names and positions.

    :param filename: path of the xyz file to create
        (overwritten if it already exists)
    :type filename: str
    :param n_atoms: number of atoms to write
    :type n_atoms: int
    :param atom_names: atom symbols, in the same order as `positions`
    :type atom_names: sequence of str
    :param positions: nuclear positions to write
    :type positions: numpy.ndarray or sequence, shape (n_atoms, 3)
    :param z_coordinate: scale factor applied to the z coordinate of each
        atom before writing it, defaults to 1.0 (unmodified structure)
    :type z_coordinate: float, optional
    """
    with open(filename, "w") as f:
        f.write(str(n_atoms) + "\n")
        f.write("XYZ file created by pyECM\n")
        for j in range(n_atoms):
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


def write_dirac_mol(
    filename, n_atoms, atom_names_dirac, atomic_numbers, positions, z_coordinate=1.0
):
    """Write a DIRAC .mol file from atom names, atomic numbers, and positions.

    :param filename: path of the .mol file to create (overwritten if it
        already exists)
    :type filename: str
    :param n_atoms: number of atoms to write
    :type n_atoms: int
    :param atom_names_dirac: per-atom DIRAC labels (element symbol + unique
        index, e.g. "H1", "H2", "O3")
    :type atom_names_dirac: sequence of str
    :param atomic_numbers: atomic number (Z) per atom, in the same order
        as `positions`
    :type atomic_numbers: sequence of int
    :param positions: nuclear positions to write
    :type positions: numpy.ndarray or sequence, shape (n_atoms, 3)
    :param z_coordinate: scale factor applied to the z coordinate of each
        atom before writing it, defaults to 1.0 (unmodified structure)
    :type z_coordinate: float, optional
    """
    with open(filename, "w") as dirac:
        dirac.write("DIRAC\n")
        dirac.write("\n")
        dirac.write("\n")
        dirac.write("C " + "{:3d}".format(n_atoms) + "  0 0         A\n")
        for j in range(n_atoms):
            dirac.write("     " + "{:3d}".format(atomic_numbers[j]) + ".     1\n")
            dirac.write(
                atom_names_dirac[j]
                + "   "
                + "{:.6f}".format(positions[j][0])
                + " "
                + "{:.6f}".format(positions[j][1])
                + " "
                + "{:.6f}".format(positions[j][2] * z_coordinate)
                + "\n"
            )
            dirac.write("LARGE BASIS base\n")
        dirac.write("FINISH")
