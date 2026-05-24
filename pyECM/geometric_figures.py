import numpy as np
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import art3d


def plot_vector(fig, orig, vector, color="blue"):
    """Plots a vector, defined by its origin and direction

    :param fig: figure (from matplotlib.pyplot)
    :type fig: matplotlib.figure.Figure
    :param orig: vector's origin coordinates
    :type orig: numpy.ndarray
    :param vector: vector's components
    :type vector: _type_
    :param color: vector's colour, defaults to "blue"
    :type color: str, optional
    :return: figure
    :rtype: matplotlib.figure.Figure
    """
    if fig.get_axes():
        ax = fig.gca()
    else:
        ax = fig.add_subplot(projection="3d")
    # ax = fig.add_subplot(projection="3d")
    # ax = fig.gca(projection="3d")
    orig = np.array(orig)
    v = np.array(vector)
    ax.quiver(orig[0], orig[1], orig[2], v[0], v[1], v[2], color=color)
    return fig


def rotation_matrix(direction):
    """Gets the matrix for objects rotations.
    The direction is taken against z-direction.

    :param direction: Final direction
    :type direction: numpy.ndarray
    :return: rotation matrix
    :rtype: numpy.ndarray
    """
    sin_angle = np.linalg.norm(direction)
    if sin_angle == 0:
        return np.identity(3)
    direction /= sin_angle
    eye = np.eye(3)
    ddt = np.outer(direction, direction)
    skew = np.array(
        [
            [0, direction[2], -direction[1]],
            [-direction[2], 0, direction[0]],
            [direction[1], -direction[0], 0],
        ],
        dtype=np.float64,
    )
    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M


def pathpatch_2d_to_3d(pathpatch, z, normal):
    #    if type(normal) is str:  # Translate strings to normal vectors
    #        index = "xyz".index(normal)
    #        normal = np.roll((1.0, 0, 0), index)

    normal /= np.linalg.norm(normal)  # Make sure the vector is normalised
    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._axlim_clip = False  # matplotlib >=3.9 compatibility
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor()  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    M = rotation_matrix(d)  # Get the rotation matrix

    pathpatch._segment3d = np.array(
        [np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts]
    )


def pathpatch_translate(pathpatch, delta):
    pathpatch._segment3d += delta


def define_plane(ax, point, normal, size=10, color="y"):
    """Defines a plane (to be plotted)

    :param ax: matplotlib axe
    :type ax: matplotlib.axes._subplots.Axes3DSubplot
    :param point: point contained by the plane
    :type point: numpy.ndarray
    :param normal: vector normal to the plane
    :type normal: numpy.ndarray
    :param size: radio size to be plotted, defaults to 10
    :type size: int, optional
    :param color: plane colour, defaults to "y"
    :type color: str, optional
    """
    p = Circle((0, 0), size, facecolor=color, alpha=0.2)
    ax.add_patch(p)
    pathpatch_2d_to_3d(p, z=0, normal=normal)
    pathpatch_translate(p, (point[0], point[1], point[2]))


def define_sphere(xCenter, yCenter, zCenter, r):
    """Defines the points that constitute a sphere, defined by
    its coordinates centers and its radio size.

    :param xCenter: X center coordinates
    :type xCenter: numpy.float64
    :param yCenter: Y center coordinates
    :type yCenter: numpy.float64
    :param zCenter: Z center coordinates
    :type zCenter: numpy.float64
    :param r: radio size
    :type r: float
    :return: points defining the sphere volume
    :rtype: numpy.ndarray
    """
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + xCenter
    y = r * y + yCenter
    z = r * z + zCenter
    return (x, y, z)


def rotation_matrix_from_vectors(vec_to_rotate, final_direction):
    """Finds the rotation matrix that aligns vec_to_rotate to final_direction

    :param vec_to_rotate: vector that will be rotated
    :type vec_to_rotate: numpy.ndarray
    :param final_direction: final direction of rotated vector
    :type final_direction: numpy.ndarray
    :return: rotation matrix
    :rtype: numpy.ndarray
    """
    a, b = (vec_to_rotate / np.linalg.norm(vec_to_rotate)).reshape(3), (
        final_direction / np.linalg.norm(final_direction)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix
