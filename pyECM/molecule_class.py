import os
import sys

import mendeleev as mendeleev
import numpy as np
from pyscf import gto

from pyECM import ccm_metrics, ecm_metric, gamma5, geometry, plotting, pyscf_wf, xyz_io
from pyECM.geometric_figures import rotation_matrix_from_vectors
from pyECM.pyscf_fc import Epv_molecule

module_path = os.path.abspath(os.path.join(".."))

if module_path not in sys.path:
    sys.path.append(module_path)


class molecula:
    """It is possible to create a molecule object from a xyz file.
    A vector can be associated to the molecule so that it is possible to align it
    with the z direction, as the virtual mirror path will be created
    by reflections in the z=0 plane.
    The vector should be obtained uploading the xzy file to
    https://csm.ouproj.org.il/molecule.

    :examples: >>> vector = np.array([-0.1807, -0.9725, -0.1469])
        >>> origin_at = np.array([0.0000, 0.200, 0.1000])
        >>> file='pyECM/data/import/CFMAR_chiral.xyz'
        >>> molecule = molecula(XYZ_file = file, direction=vector, origin=origin_at)
    :param preloaded_molecule: tuple (n_atoms, atoms, direction, bonds)
        for building the molecule without reading a xyz file, defaults to None
    :type preloaded_molecule: tuple, optional
    :param figure: matplotlib figure where the molecule will be plotted,
        required only if any plot_* method will be used, defaults to None
    :type figure: matplotlib.figure.Figure, optional
    :param origin: point (or atom name) used as origin of coordinates,
        defaults to None (uses the first atom's position)
    :type origin: numpy.ndarray or str, optional
    :param XYZ_file: path to the xyz file with the chiral structure, defaults to None
    :type XYZ_file: str, optional
    :param XYZ_achiral_file: path to the xyz file with the achiral
        (symmetric) reference structure, defaults to None
    :type XYZ_achiral_file: str, optional
    :param direction: vector defining the molecule orientation with respect to
        the nearest achiral symmetric structure,
        used to align it with the z axis, defaults to None
    :type direction: numpy.ndarray, optional
    :param kwargs: extra plotting options, stored in self.options and
        evaluated by plot_options()
    """

    def __init__(
        self,
        preloaded_molecule=None,
        figure=None,
        origin=None,
        XYZ_file=None,
        XYZ_achiral_file=None,
        direction=None,
        **kwargs
    ):
        self.fig = figure
        self.options = kwargs
        self.origin = origin
        self.bohrtoang = 0.529177249

        if preloaded_molecule is not None:
            self.preloaded_molecule = preloaded_molecule
            self.n_atoms, self.atoms, self.direction, self.bonds = preloaded_molecule

        if XYZ_file is not None:
            self.XYZ_file = XYZ_file
            self.load_from_xyz(filename=self.XYZ_file)
            self.direction = direction

        if XYZ_achiral_file is not None:
            self.XYZ_achiral_file = XYZ_achiral_file
            self.load_from_xyz(filename=self.XYZ_achiral_file, achiral=True)

        else:
            self.XYZ_achiral_file = None

        # self.validar()
        self.atoms_positions()
        self.reference_coordinate()

        if self.origin is not None:
            self.origin_on_atom()
        else:
            self.origin = np.array(
                [self.positions[0][0], self.positions[0][1], self.positions[0][2]]
            )

    def atoms_positions(self):
        """Builds self.positions (and self.positions_achiral, if an achiral
        xyz file was loaded) as (n_atoms, 3) arrays from self.atoms."""
        self.positions = geometry.build_positions(self.n_atoms, self.atoms)

        if self.XYZ_achiral_file is not None:
            self.positions_achiral = geometry.build_positions(
                self.n_atoms, self.achiral_atoms
            )

    def reference_coordinate(self):
        """Sets self.central_point from self.origin: either the position of
        the named atom (if origin is a string) or the given point directly
        (if origin is a numpy array)."""
        central_point = geometry.central_point_from_origin(
            self.n_atoms, self.positions, self.atoms[4], self.origin
        )
        if central_point is not None:
            self.central_point = central_point

    def origin_on_atom(self):
        """Shifts self.positions so that self.origin becomes the new
        coordinate origin, and stores the original central point in
        self.central_point_coordinates."""
        self.positions, self.central_point_coordinates = (
            geometry.shift_positions_to_origin(
                self.n_atoms, self.positions, self.atoms[4], self.origin
            )
        )

    def rotate_to_align_with_z(self):
        """Rotates the molecule so that its nearest symmetric
        structure lies in the z=0 plane."""
        z_direction = np.array([0, 0, 1])
        if z_direction.all() != self.direction.all():
            matrix = rotation_matrix_from_vectors(self.direction, z_direction)
            for i in range(self.n_atoms):
                self.positions[i] = matrix @ self.positions[i]
            self.direction = matrix @ self.direction

    def plot_dipole(self):
        """Plots the molecule dipole (if direction is replaced
        by the molecule dipole)."""
        plotting.plot_dipole(self)

    def plot_plane(self):
        """Plots the plane normal to the molecule direction."""
        plotting.plot_plane(self)

    def plot_sphere(self):
        """Plots the nuclei as spheres."""
        plotting.plot_sphere(self)

    def plot_bonds(self):
        """Plot the molecule bonds."""
        plotting.plot_bonds(self)

    def plot_options(self):
        """Options plots."""
        plotting.plot_options(self)

    def save_xyz(self, filename="MOL"):
        """Saves the molecule in xyz format

        :param filename: xyz file name, defaults to "MOL"
        :type filename: str, optional
        """
        atoms_name_xyz = [
            xyz_io.atom_symbol(self.atoms[4][j]) for j in range(self.n_atoms)
        ]
        xyz_io.write_xyz(filename, self.n_atoms, atoms_name_xyz, self.positions)

    def load_from_xyz(self, filename="MOL", achiral=False):
        """Loads atomic positions and names from a xyz file into self.atoms
        (or self.achiral_atoms, if achiral=True).

        :param filename: xyz file name, defaults to "MOL"
        :type filename: str, optional
        :param achiral: if True, stores the result in self.achiral_atoms
            instead of self.atoms/self.n_atoms, defaults to False
        :type achiral: bool, optional
        """

        number_atoms, coord_x, coord_y, coord_z, colors, names = xyz_io.read_xyz(
            filename
        )

        if achiral:
            self.achiral_atoms = (coord_x, coord_y, coord_z, colors, names)
        else:
            self.n_atoms = number_atoms
            self.atoms = (coord_x, coord_y, coord_z, colors, names)

    def export_xyz(
        self,
        prefix_name="MOL",
        DIRAC=False,
        folder=None,
        z_coordinate=1.00,
        achiral=None,
    ):
        """Generates the xyz files for the chiral molecule and
        the nearest symmetric structure.

        :param prefix_name: prefix name for xyz files, defaults to "MOL"
        :type prefix_name: str, optional
        :param DIRAC: create DIRAC mol files, defaults to False
        :type DIRAC: bool, optional
        :param folder: directory where saving the files, defaults to None
        :type folder: str, optional
        :param z_coordinate: scale factor applied to the z coordinate of the
            chiral structure, defaults to 1.00 (unmodified structure)
        :type z_coordinate: float, optional
        :param achiral: path to an existing xyz file to use as the achiral
            reference (copied as-is). If None, the achiral structure is
            derived from the chiral one by zeroing its z coordinate,
            defaults to None
        :type achiral: str, optional
        """
        atoms_name_xyz = [
            xyz_io.atom_symbol(self.atoms[4][j]) for j in range(self.n_atoms)
        ]
        atoms_name_dirac = [
            str(self.atoms[4][j]) + str(j + 1) for j in range(self.n_atoms)
        ]
        atoms_Z = [getattr(mendeleev, name).atomic_number for name in atoms_name_xyz]

        # We always need the nearest assymetric structure
        filename = folder + prefix_name + "_" + "0.00.xyz"
        # Remove xyz file if exist
        try:
            os.remove(filename)
        except OSError:
            pass

        if achiral is None:
            row_0 = [self.n_atoms, "", "", ""]
            row_1 = ["XYZ file", "", "", ""]
            rows = np.vstack([row_0, row_1])
            for j in range(self.n_atoms):
                new_line = np.array(
                    [
                        atoms_name_xyz[j],
                        self.positions[j][0],
                        self.positions[j][1],
                        self.positions[j][2] * 0,
                    ]
                )
                rows = np.vstack([rows, new_line])

            with open(filename, "ab") as f:
                np.savetxt(f, rows, fmt="%s")
        else:
            import shutil

            shutil.copyfile(achiral, folder + "/" + prefix_name + "_0.00.xyz")

        # Export the (rotated) chiral molecule
        filename = folder + prefix_name + "_" + "{:.2f}".format(z_coordinate) + ".xyz"
        try:
            os.remove(filename)
        except OSError:
            pass
        xyz_io.write_xyz(
            filename, self.n_atoms, atoms_name_xyz, self.positions, z_coordinate
        )

        if DIRAC:
            filename_DIRAC = (
                folder
                + "D22_"
                + prefix_name
                + "_"
                + "{:.2f}".format(z_coordinate)
                + ".mol"
            )
            xyz_io.write_dirac_mol(
                filename_DIRAC,
                self.n_atoms,
                atoms_name_dirac,
                atoms_Z,
                self.positions,
                z_coordinate,
            )

    def xyz_mirror_path(
        self,
        prefix_name="MOL",
        DIRAC=False,
        folder=None,
        lim_inf=0.20,
        lim_sup=1.00,
        points=10,
    ):
        """Generates the molecules in the virtual mirror path, which is defined
        by reflecting the molecule in the plane z=0.

        :param prefix_name: prefix name for xyz files, defaults to "MOL"
        :type prefix_name: str, optional
        :param DIRAC: create DIRAC mol files, defaults to False
        :type DIRAC: bool, optional
        :param folder: directory where saving the files, defaults to None
        :type folder: str, optional
        :param lim_inf: minimum z-rate, defaults to 0.0
        :type lim_inf: float, optional
        :param lim_sup: maximum z-rate, defaults to 1.0
        :type lim_sup: float, optional
        :param points: number of z-rates defining the path, defaults to 10.0
        :type points: float, optional
        """

        if points is None:
            points = int((lim_sup - lim_inf) / 0.05 + 1)

        for i in np.linspace(lim_inf, lim_sup, points):
            self.export_xyz(prefix_name, DIRAC, folder, i)

    def CCM(self, z_coordinate=1.00, path=False):
        """Calculates the CCM for the molecule.

        :param z_coordinate: scale factor applied to the z coordinate,
            defaults to 1.00
        :type z_coordinate: float, optional
        :param path: if True, returns the results instead of storing them
            in self (used when called repeatedly from CCM_on_path),
            defaults to False
        :type path: bool, optional
        :return: NORM1, CCM1, NORM2, CCM2
        :rtype: float(s)
        """

        positions_achiral = (
            None if self.XYZ_achiral_file is None else self.positions_achiral
        )

        Norm_1, CCM_1, Norm_2, CCM_2 = ccm_metrics.ccm(
            self.positions,
            self.atoms[4],
            positions_achiral=positions_achiral,
            z_coordinate=z_coordinate,
        )

        if path is False:
            self.Norm1 = Norm_1
            self.CCM1 = CCM_1
            self.Norm2 = Norm_2
            self.CCM2 = CCM_2
        elif path is True:
            return Norm_1, CCM_1, Norm_2, CCM_2

    def CCM_on_path(self, lim_inf=0.20, lim_sup=1.00, points=10):
        """Calculates the CCM for the molecules in the virtual mirror path.

        :param lim_inf: minimum z-rate, defaults to 0.20
        :type lim_inf: float, optional
        :param lim_sup: maximum z-rate, defaults to 1.00
        :type lim_sup: float, optional
        :param points: number of z-rates where calculating the CCM, defaults to 10
        :type points: int, optional
        :return: z-rates, NORMs1, CCMs1, NORMs2, CCMs2
        :rtype: numpy.ndarray(s)
        """
        z_rate = np.zeros(points)
        CCMs_1 = np.zeros(points)
        Norms_1 = np.zeros(points)
        CCMs_2 = np.zeros(points)
        Norms_2 = np.zeros(points)
        index = 0
        for j in np.linspace(lim_inf, lim_sup, points):
            z_rate[index] = j
            Norms_1[index], CCMs_1[index], Norms_2[index], CCMs_2[index] = self.CCM(
                z_coordinate=j, path=True
            )
            index = index + 1

        return z_rate, Norms_1, CCMs_1, Norms_2, CCMs_2

    def pySCF_WF(
        self,
        name=None,
        cartesian=False,
        z_coordinate=1.00,
        gto_dict=None,
        method_dict=None,
    ):
        """Obtain the wave function with the pySCF code, storing the result(s)
        as attributes of self (e.g. self.NR_all_MO, self.x2c_MO,
        self.rel_MO_Lo/So, depending on which methods were requested).
        NR, X2C and 4c are independent and can be combined freely in
        a single call.

        :param name: name of the xyz molecule file,
            including its directory, defaults to None
        :type name: str, optional
        :param cartesian: use cartesian basis set, defaults to False
        :type cartesian: bool, optional
        :param z_coordinate: scale factor applied to the z coordinate of the
            structure whose WF is computed, defaults to 1.00
        :type z_coordinate: float, optional
        :param gto_dict: options for Gaussian Type Orbitals (basis, charge,
            spin, verbose), defaults to None
        :type gto_dict: dict, optional
        :param method_dict: which WF method(s) to compute and their options
            (NR, X2C, fourcomp, DFT, debug, cvalue). Keys not given fall
            back to defaults (NR=True, the rest False/0), defaults to None
        :type method_dict: dict, optional
        :raises NotImplementedError: if fourcomp and DFT are both requested
        """

        # Define default values for keys in gto_dict
        if gto_dict is None:
            gto_dict = {}
        gto_dict.setdefault("basis", "sto-6g")
        gto_dict.setdefault("charge", None)
        gto_dict.setdefault("spin", None)
        gto_dict.setdefault("verbose", 0)

        self.gto_dict = gto_dict

        # Define default values for keys in method_dict
        if method_dict is None:
            method_dict = {}
        NR = method_dict.get("NR", True)
        fourcomp = method_dict.get("fourcomp", False)
        X2C = method_dict.get("X2C", False)
        DFT = method_dict.get("DFT", False)
        debug = method_dict.get("debug", 0)
        cvalue = method_dict.get("cvalue", 137.03599967994)

        if fourcomp is not False and DFT is not False:
            raise NotImplementedError("4c-DFT not available yet.")

        mol_chiral = self._build_mol_chiral(name, z_coordinate, cartesian)

        self.AO_number = mol_chiral.nao

        if NR:
            (
                self.NR_Noccupied_MO_alpha,
                self.NR_Noccupied_MO_beta,
                self.NR_all_MO,
                self.NR_occupied_MO,
                self.NR_pyscf,
                _,
            ) = pyscf_wf.compute_NR_WF(mol_chiral, DFT=DFT, debug=debug)

        if X2C:
            (
                self.x2c_MO,
                self.x2c_occup_MO,
                self.x2c_Nalphaoccupied_MO,
                self.x2c_Nbetaoccupied_MO,
                self.x2c_energy,
                _,
            ) = pyscf_wf.compute_X2C_WF(mol_chiral, DFT=DFT, debug=debug)

        if fourcomp:
            (
                self.n4c,
                self.rel_nmo,
                self.rel_MO_Lo,
                self.rel_MO_So,
                self.rel_Noccupied_MO,
                self.rel_energy,
                self.rel_pyscf,
                _,
            ) = pyscf_wf.compute_4c_WF(mol_chiral, cvalue, debug=debug)

    def _build_mol_chiral(self, name, z_coordinate, cartesian):
        """Build the pyscf.gto.Mole for the chiral structure, scaled on
        the z-axis by the given z_coordinate, using self.gto_dict (set by pySCF_WF).

        :param name: name of the xyz molecule file, including its directory
        :type name: str
        :param z_coordinate: scale factor applied to the z coordinate
        :type z_coordinate: float
        :param cartesian: use cartesian basis set
        :type cartesian: bool
        :return: the built molecule
        :rtype: pyscf.gto.Mole
        """
        mol_chiral = gto.M(
            atom=name + "_" + "{:.2f}".format(z_coordinate) + ".xyz",
            max_memory=5000.0,
            **self.gto_dict
        )
        if cartesian:
            mol_chiral, _ = mol_chiral.to_uncontracted_cartesian_basis()
        return mol_chiral

    def _build_mol_achiral(self, name, cartesian):
        """Build the pyscf.gto.Mole for the achiral reference structure,
        using self.gto_dict (set by pySCF_WF).

        :param name: name of the xyz molecule file, including its directory
        :type name: str
        :param cartesian: use cartesian basis set
        :type cartesian: bool
        :return: the built molecule
        :rtype: pyscf.gto.Mole
        """
        mol_achiral = gto.M(atom=name + "_0.00.xyz", max_memory=5000.0, **self.gto_dict)
        if cartesian:
            mol_achiral, _ = mol_achiral.to_uncontracted_cartesian_basis()
        return mol_achiral

    def ECM(
        self,
        name=None,
        cartesian=False,
        z_coordinate=1.00,
        path=False,
        method_dict=None,
    ):
        """Calculate ECM in a certain structure

        :param name: name of the xyz molecule file,
            including its directory, defaults to None
        :type name: str, optional
        :param cartesian: use cartesian basis set, defaults to False
        :type cartesian: bool, optional
        :param z_coordinate: variable for ECM on path, defaults to 1.00
        :type z_coordinate: float, optional
        :param path: True for ECM on path, defaults to False
        :type path: bool, optional
        :param method_dict: Set method for WF calculation,
            previously calculated, defaults to None
        :type method_dict: dict, optional
        :raises AttributeError: if the wave function required by
            method_dict was not computed first with pySCF_WF
        :return: ECM_NR, ECM_X2C, ECM_4c (only when path=True)
        :rtype: float(s)
        """

        # Define default values for keys in method_dict
        if method_dict is None:
            method_dict = {}
        NR = method_dict.get("NR", True)
        fourcomp = method_dict.get("fourcomp", False)
        X2C = method_dict.get("X2C", False)
        debug = method_dict.get("debug", 0)
        cvalue = method_dict.get("cvalue", 137.03599967994)

        if NR and not hasattr(self, "NR_all_MO"):
            raise AttributeError(
                "The non-relativistic wave function is not defined. "
                "Run pySCF_WF(method_dict={'NR': True, ...}) first."
            )
        if X2C and not hasattr(self, "x2c_MO"):
            raise AttributeError(
                "The X2C wave function is not defined. "
                "Run pySCF_WF(method_dict={'X2C': True, ...}) first."
            )
        if fourcomp and not hasattr(self, "rel_MO_Lo"):
            raise AttributeError(
                "The four-component wave function is not defined. "
                "Run pySCF_WF(method_dict={'fourcomp': True, ...}) first."
            )

        ECM_NR = None
        ECM_X2C = None
        ECM_4c = None
        ECM_NR_molcontr_alpha = []
        ECM_NR_molcontr_beta = []
        ECM_NR_molcontr = []
        ECM_X2C_molcontr = []
        ECM_4c_molcontr = []

        mol_chiral = self._build_mol_chiral(name, z_coordinate, cartesian)
        mol_achiral = self._build_mol_achiral(name, cartesian)

        mol_super = mol_chiral + mol_achiral

        if NR:
            (
                ECM_NR,
                ECM_NR_molcontr_alpha,
                ECM_NR_molcontr_beta,
                ECM_NR_molcontr,
            ) = ecm_metric.compute_ECM_NR(
                mol_chiral,
                mol_achiral,
                mol_super,
                self.NR_all_MO,
                self.NR_Noccupied_MO_alpha,
                self.NR_Noccupied_MO_beta,
                debug=debug,
            )

        if X2C:
            ECM_X2C, ECM_X2C_molcontr = ecm_metric.compute_ECM_X2C(
                mol_chiral,
                mol_achiral,
                mol_super,
                self.x2c_MO,
                self.x2c_occup_MO,
                self.x2c_Nalphaoccupied_MO,
                self.x2c_Nbetaoccupied_MO,
                debug=debug,
            )

        if fourcomp:
            ECM_4c, ECM_4c_molcontr = ecm_metric.compute_ECM_4c(
                mol_chiral,
                mol_achiral,
                self.n4c,
                self.rel_MO_Lo,
                self.rel_MO_So,
                self.rel_Noccupied_MO,
                cvalue,
                debug=debug,
            )

        if path is False:
            if NR:
                self.ECM_NR = ECM_NR
                self.ECM_NR_molcontr_alpha = ECM_NR_molcontr_alpha
                self.ECM_NR_molcontr_beta = ECM_NR_molcontr_beta
                self.ECM_NR_molcontr = ECM_NR_molcontr
            if X2C:
                self.ECM_X2C = ECM_X2C
                self.ECM_X2C_molcontr = ECM_X2C_molcontr
            if fourcomp:
                self.ECM_4c = ECM_4c
                self.ECM_4c_molcontr = ECM_4c_molcontr
        elif path is True:
            return ECM_NR, ECM_X2C, ECM_4c

    def ECM_on_path(
        self,
        name=None,
        lim_inf=0.20,
        lim_sup=1.00,
        points=10,
        cartesian=False,
        method_dict=None,
        gto_dict=None,
    ):
        """Calculate ECM in the virtual mirror path

        :param name: name of the xyz molecule file,
            including its directory, defaults to None
        :type name: str, optional
        :param lim_inf: minimum z-rate, defaults to 0.20
        :type lim_inf: float, optional
        :param lim_sup: maximum z-rate, defaults to 1.00
        :type lim_sup: float, optional
        :param points: number of z-rates where calculating the ECM, defaults to 10
        :type points: int, optional
        :param method_dict: Set method for WF calculation
            (previously calculated), defaults to None
        :type method_dict: dict, optional
        :param gto_dict: Set options for Gaussian Type Orbitals, defaults to None
        :type gto_dict: dict, optional
        :return: z-rates, ECMs(NR), molecular orbital contributions
            to ECMs(NR), ECMs(4c)
        :rtype: numpy.ndarray(s)
        """

        z_rate = np.zeros(points)
        ECMs_NR = np.zeros(points)
        ECMs_x2c = np.zeros(points)
        ECMs_4c = np.zeros(points)

        index = 0

        for j in np.linspace(lim_inf, lim_sup, points):
            z_rate[index] = j

            self.pySCF_WF(
                name,
                z_coordinate=j,
                gto_dict=gto_dict,
                method_dict=method_dict,
                cartesian=cartesian,
            )

            ECMs_NR[index], ECMs_x2c[index], ECMs_4c[index] = self.ECM(
                name,
                method_dict=method_dict,
                z_coordinate=j,
                path=True,
                cartesian=cartesian,
            )
            index = index + 1

        return z_rate, ECMs_NR, ECMs_x2c, ECMs_4c

    def gamma5(
        self, name=None, cartesian=False, z_coordinate=1.00, method_dict=None, debug=0
    ):
        """Calculate the Gamma5 expectation value. Requires that the
        four-component wave function was already computed with
        pySCF_WF(method_dict={'fourcomp': True, ...}).

        :param name: name of the xyz molecule file,
            including its directory, defaults to None
        :type name: str, optional
        :param cartesian: use cartesian basis set, defaults to False
        :type cartesian: bool, optional
        :param z_coordinate: scale factor applied to the z coordinate of the
            structure, defaults to 1.00
        :type z_coordinate: float, optional
        :param method_dict: options for the calculation (debug, cvalue),
            defaults to None
        :type method_dict: dict, optional
        :param debug: debug level, overridden by method_dict["debug"]
            if given, defaults to 0
        :type debug: int, optional
        :raises AttributeError: if the four-component wave function is not defined
        :return: Gamma5 expectation value
        :rtype: float
        """

        if not hasattr(self, "rel_MO_Lo"):
            raise AttributeError(
                "The four-component wave function is not defined within the class. "
            )

        # Define default values for keys in method_dict
        if method_dict is None:
            method_dict = {}
        debug = method_dict.get("debug", 0)
        cvalue = method_dict.get("cvalue", 137.03599967994)

        mol_chiral = self._build_mol_chiral(name, z_coordinate, cartesian)

        return gamma5.compute_gamma5(
            mol_chiral,
            self.n4c,
            self.rel_Noccupied_MO,
            self.rel_MO_Lo,
            self.rel_MO_So,
            cvalue,
            self.rel_energy,
            debug=debug,
        )

    def Epv(
        self,
        name=None,
        cartesian=False,
        z_coordinate=1.00,
        method_dict=None,
        debug=0,
        dm=None,
    ):
        """Calculate the parity-violating (PV) energy contribution for each
        atom and occupied orbital, storing the result in self.Epv_expval.
        Requires that the four-component wave function was already computed
        with pySCF_WF(method_dict={'fourcomp': True, ...}).

        :param name: name of the xyz molecule file, including its directory.
            Currently unused (kept for API consistency with ECM/gamma5),
            defaults to None
        :type name: str, optional
        :param cartesian: use cartesian basis set. Currently unused,
            defaults to False
        :type cartesian: bool, optional
        :param z_coordinate: scale factor applied to the z coordinate.
            Currently unused, defaults to 1.00
        :type z_coordinate: float, optional
        :param method_dict: options for the calculation. Currently unused,
            defaults to None
        :type method_dict: dict, optional
        :param debug: debug level. Currently unused, defaults to 0
        :type debug: int, optional
        :param dm: density matrix (AO or MO basis) to contract the PV
            operator with. If None, uses the reference DHF orbitals directly,
            defaults to None
        :type dm: numpy.ndarray, optional
        :raises AttributeError: if the four-component wave function is not defined
        """

        if not hasattr(self, "rel_MO_Lo"):
            raise AttributeError(
                "The four-component wave function is not defined within the class. "
            )

        self.Epv_expval = Epv_molecule(self.rel_pyscf.mol, self.rel_pyscf, dm)
