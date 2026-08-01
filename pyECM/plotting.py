"""Plotting utilities for pyECM.molecula objects."""

import numpy as np

from pyECM.geometric_figures import define_plane, define_sphere, plot_vector


def _get_axes(mol):
    """Devuelve los ejes 3D actuales de mol.fig, creándolos si hace falta."""
    if mol.fig.get_axes():
        return mol.fig.gca()
    return mol.fig.add_subplot(projection="3d")


def plot_dipole(mol):
    """Plots the molecule dipole (if direction is replaced
    by the molecule dipole)."""
    point = np.array([0, 0, 0])
    dipolo = np.array([mol.direction[0], mol.direction[1], mol.direction[2]])
    plot_vector(mol.fig, point, dipolo)


def plot_plane(mol):
    """Plots the plane normal to the molecule direction."""
    point = np.array([0.0, 0.0, 0.0])
    dipolo = np.array([mol.direction[0], mol.direction[1], mol.direction[2]])
    ax = _get_axes(mol)
    define_plane(ax, point, dipolo, size=1)


def plot_sphere(mol):
    """Plots the nuclei as spheres."""
    r = 0.05
    ax = _get_axes(mol)
    for i in range(mol.nro_atoms):
        xs, ys, zs = define_sphere(
            mol.positions[i][0], mol.positions[i][1], mol.positions[i][2], r
        )
        ax.plot_wireframe(xs, ys, zs, color=mol.atoms[3][i])


def plot_enlaces(mol):
    """Plot the molecule bonds."""
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
    """Options plots."""
    ax = _get_axes(mol)
    ax.set_axis_on()
    for i in mol.opciones:
        exec(mol.opciones[i], {"ax": ax, "self": mol, "np": np, "__builtins__": {}})
