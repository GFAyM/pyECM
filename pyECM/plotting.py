"""Plotting utilities for pyECM.molecula objects.

Each function receives the mol object  as a parameter, so that they can be tested
independently from the mol class. They all assume that mol.fig was provided.
"""

import numpy as np

from pyECM.geometric_figures import define_plane, define_sphere, plot_vector


def _get_axes(mol):
    """Return the current 3D axes of mol.fig, creating them if needed.

    :param mol: molecule whose figure's axes are retrieved (or created)
    :type mol: pyECM.molecule_class.molecula
    :return: existing or newly created 3D axes of mol.fig
    :rtype: mpl_toolkits.mplot3d.axes3d.Axes3D
    """
    if mol.fig.get_axes():
        return mol.fig.gca()
    return mol.fig.add_subplot(projection="3d")


def plot_dipole(mol):
    """Plot the molecule dipole as a vector from the origin.

    If mol.direction was replaced by the molecule's actual dipole moment,
    plots that; otherwise plots whatever vector mol.direction currently
    holds.

    :param mol: molecule whose direction/dipole vector is plotted
    :type mol: pyECM.molecule_class.molecula
    """
    point = np.array([0, 0, 0])
    dipolo = np.array([mol.direction[0], mol.direction[1], mol.direction[2]])
    plot_vector(mol.fig, point, dipolo)


def plot_plane(mol):
    """Plot the plane normal to the molecule direction, centered at the origin.

    :param mol: molecule whose direction defines the plane's normal vector
    :type mol: pyECM.molecule_class.molecula
    """
    point = np.array([0.0, 0.0, 0.0])
    dipolo = np.array([mol.direction[0], mol.direction[1], mol.direction[2]])
    ax = _get_axes(mol)
    define_plane(ax, point, dipolo, size=1)


def plot_sphere(mol):
    """Plot each nucleus of the molecule as a small wireframe sphere.

    Colored according to mol.atoms[3] (the atom color list).

    :param mol: molecule whose nuclei are plotted
    :type mol: pyECM.molecule_class.molecula
    """
    r = 0.05
    ax = _get_axes(mol)
    for i in range(mol.n_atoms):
        xs, ys, zs = define_sphere(
            mol.positions[i][0], mol.positions[i][1], mol.positions[i][2], r
        )
        ax.plot_wireframe(xs, ys, zs, color=mol.atoms[3][i])


def plot_bonds(mol):
    """Plot the molecule bonds as straight lines.

    Drawn between the bonded atoms listed in mol.bonds.

    :param mol: molecule whose bonds are plotted. Requires mol.bonds, an (n_bonds, 2)
        array of atom index pairs
    :type mol: pyECM.molecule_class.molecula
    """
    ax = _get_axes(mol)
    for i in range(len(mol.bonds)):
        Ax = float(np.array(mol.positions)[mol.bonds[i, 0], 0])
        Ay = float(np.array(mol.positions)[mol.bonds[i, 0], 1])
        Az = float(np.array(mol.positions)[mol.bonds[i, 0], 2])
        Bx = float(np.array(mol.positions)[mol.bonds[i, 1], 0])
        By = float(np.array(mol.positions)[mol.bonds[i, 1], 1])
        Bz = float(np.array(mol.positions)[mol.bonds[i, 1], 2])
        ax.plot([Ax, Bx], [Ay, By], zs=[Az, Bz])


def plot_options(mol):
    """Apply arbitrary matplotlib customization code stored in mol.options.

    E.g. axis labels, title. Each value in mol.options is a string of
    Python code, executed with a restricted namespace exposing only 'ax'
    (the current axes), 'self' (mol, kept for backward compatibility with
    strings referencing 'self.fig', etc.) and 'np'; builtins are disabled
    to avoid arbitrary code execution (e.g. file access, imports) through
    this mechanism.

    :param mol: molecule whose mol.options dict of code strings is executed
    :type mol: pyECM.molecule_class.molecula
    """
    ax = _get_axes(mol)
    ax.set_axis_on()
    for i in mol.options:
        exec(mol.options[i], {"ax": ax, "self": mol, "np": np, "__builtins__": {}})
