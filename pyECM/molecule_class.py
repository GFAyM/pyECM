import os
import sys
import time


import mendeleev as mendeleev
import numpy as np
from numpy import matmul as mm
from numpy import transpose as tp
from pyscf import gto, lib, scf
from pyscf.lib.misc import light_speed
from scipy.linalg import fractional_matrix_power as matrix_power
from pyECM.geometric_figures import define_plane, define_sphere
from pyECM.geometric_figures import plot_vector, rotation_matrix_from_vectors
from pyECM.pyscf_fc import get_ovlp_AUCAR, Epv_molecule

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
        >>> molecule = molecula(XYZ_file = file, directio=vector, origen=origin_at)
    :param preloaded_molecule: _description_, defaults to None
    :type preloaded_molecule: _type_, optional
    :param figure: _description_, defaults to None
    :type figure: _type_, optional
    :param origen: _description_, defaults to None
    :type origen: _type_, optional
    :param XYZ_file: _description_, defaults to None
    :type XYZ_file: _type_, optional
    :param direction: _description_, defaults to None
    :type direction: _type_, optional
    """

    def __init__(
        self,
        preloaded_molecule=None,
        figure=None,
        origen=None,
        XYZ_file=None,
        XYZ_achiral_file=None,
        direction=None,
        **kwargs
    ):
        self.fig = figure
        self.opciones = kwargs
        self.origen = origen
        self.bohrtoang = 0.529177249

        if preloaded_molecule is not None:
            self.preloaded_molecule = preloaded_molecule
            self.nro_atoms, self.atoms, self.direction, self.bonds = preloaded_molecule

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
        self.coordenadas_central()

        if self.origen is not None:
            self.origin_on_atom()
        else:
            self.origen = np.array(
                [self.positions[0][0], self.positions[0][1], self.positions[0][2]]
            )

    # def validar(self):
    #    if not (type(self.nro_atoms) == int and type(self.atoms) == tuple):
    #        raise TypeError("Type error on nro_atoms and/or atoms variable/s")

    def atoms_positions(self):
        """Defines nuclei positions."""
        positions = []
        for i in range(self.nro_atoms):
            positions.append([self.atoms[0][i], self.atoms[1][i], self.atoms[2][i]])
        self.positions = np.array(positions)

        if self.XYZ_achiral_file is not None:
            positions_achiral = []
            for i in range(self.nro_atoms):
                positions_achiral.append(
                    [
                        self.achiral_atoms[0][i],
                        self.achiral_atoms[1][i],
                        self.achiral_atoms[2][i],
                    ]
                )
            self.positions_achiral = np.array(positions_achiral)

    def coordenadas_central(self):
        """Defines central point."""
        if self.origen is None:
            pass
        elif isinstance(self.origen, str):
            for i in range(self.nro_atoms):
                if self.atoms[4][i] == self.origen:
                    self.punto_central = np.array(
                        [
                            self.positions[i][0],
                            self.positions[i][1],
                            self.positions[i][2],
                        ]
                    )
        elif isinstance(self.origen, np.ndarray):
            self.punto_central = self.origen

    def origin_on_atom(self):
        """Gets central point coordinates."""
        if isinstance(self.origen, str):
            for i in range(self.nro_atoms):
                if self.atoms[4][i] == self.origen:
                    coordenadas_punto_central = np.array(
                        [
                            self.positions[i][0],
                            self.positions[i][1],
                            self.positions[i][2],
                        ]
                    )
        if isinstance(self.origen, np.ndarray):
            coordenadas_punto_central = self.origen
        new_positions = np.zeros(3)
        for i in range(self.nro_atoms):
            new_line = self.positions[i] - coordenadas_punto_central
            new_positions = np.vstack([new_positions, new_line])
        new_positions = np.delete(new_positions, (0), axis=0)  # Borro fila 1 de ceros
        self.positions = new_positions
        self.coordenadas_punto_central = coordenadas_punto_central

    def rotate_to_align_with_z(self):
        """Rotates the molecule so that its nearest symmetric
        structure lies in the z=0 plane."""
        z_direction = np.array([0, 0, 1])
        if z_direction.all() != self.direction.all():
            matrix = rotation_matrix_from_vectors(self.direction, z_direction)
            for i in range(self.nro_atoms):
                self.positions[i] = matrix @ self.positions[i]
            self.direction = matrix @ self.direction

    def plot_dipolo(self):
        """Plots the molecule dipole (if direction is replaced
        by the molecule dipole)."""
        point = np.array([0, 0, 0])
        dipolo = np.array([self.direction[0], self.direction[1], self.direction[2]])
        plot_vector(self.fig, point, dipolo)

    def plot_plano(self):
        """Plots the plane normal to the molecule direction."""
        point = np.array([0.0, 0.0, 0.0])
        dipolo = np.array([self.direction[0], self.direction[1], self.direction[2]])
        if self.fig.get_axes():
            ax = self.fig.gca()
        else:
            ax = self.fig.add_subplot(projection="3d")
        define_plane(ax, point, dipolo, size=1)

    def plot_sphere(self):
        """Plots the nuclei as spheres."""
        r = 0.05
        if self.fig.get_axes():
            ax = self.fig.gca()
        else:
            ax = self.fig.add_subplot(projection="3d")

        for i in range(self.nro_atoms):
            (xs, ys, zs) = define_sphere(
                self.positions[i][0], self.positions[i][1], self.positions[i][2], r
            )
            ax.plot_wireframe(xs, ys, zs, color=self.atoms[3][i])

    def plot_enlaces(self):
        """Plot the molecule bonds."""
        for i in range(len(self.bonds)):
            Ax = float(np.array(self.positions)[self.bonds[i, 0], 0])
            Ay = float(np.array(self.positions)[self.bonds[i, 0], 1])
            Az = float(np.array(self.positions)[self.bonds[i, 0], 2])
            Bx = float(np.array(self.positions)[self.bonds[i, 1], 0])
            By = float(np.array(self.positions)[self.bonds[i, 1], 1])
            Bz = float(np.array(self.positions)[self.bonds[i, 1], 2])
            # draw diagonal line from (Ax,Ay,Az) to (Bx,By,Bz)
            # ax = self.fig.gca(projection="3d")
            if self.fig.get_axes():
                ax = self.fig.gca()
            else:
                ax = self.fig.add_subplot(projection="3d")

            ax.plot([Ax, Bx], [Ay, By], zs=[Az, Bz])

    def plot_opciones(self):
        """Options plots."""
        if self.fig.get_axes():
            ax = self.fig.gca()
        else:
            ax = self.fig.add_subplot(projection="3d")
        ax.set_axis_on()
        for i in self.opciones:
            exec(self.opciones[i])

    def save_xyz(self, filename="MOL"):
        """Saves the molecule in xyz format

        :param filename: xyz file name, defaults to "MOL"
        :type filename: str, optional
        """
        atoms_name_xyz = []
        atoms_Z = []
        for j in range(self.nro_atoms):
            atom_name_xyz = "".join(
                [i for i in str(self.atoms[4][j]) if not i.isdigit()]
            )
            atoms_name_xyz.append(atom_name_xyz)
            atoms_Z.append(eval("mendeleev." + atom_name_xyz + ".atomic_number"))

        # filas =  np.zeros(4)
        fila_0 = [self.nro_atoms, "", "", ""]
        fila_1 = ["XYZ file", "", "", ""]
        filas = np.vstack([fila_0, fila_1])
        for j in range(self.nro_atoms):
            new_line = np.array(
                [
                    self.atoms[4][j],
                    self.positions[j][0],
                    self.positions[j][1],
                    self.positions[j][2],
                ]
            )
            filas = np.vstack([filas, new_line])
        # with open(filename, "ab") as f:
        #    np.savetxt(f, filas, fmt="%s")

        with open(filename, "w") as f:
            f.write(str(self.nro_atoms) + "\n")
            f.write("XYZ file created by pyECM\n")
            for j in range(self.nro_atoms):
                f.write(
                    atoms_name_xyz[j]
                    + "   "
                    + "{:.6f}".format(self.positions[j][0])
                    + " "
                    + "{:.6f}".format(self.positions[j][1])
                    + " "
                    + "{:.6f}".format(self.positions[j][2])
                    + "\n"
                )

    def load_from_xyz(self, filename="MOL", achiral=False):
        """Loads the molecule from a xyz file.

        :param filename: xyz file name, defaults to "MOL"
        :type filename: str, optional
        """

        # Check for empty lines in xyz file.
        xyz_check = open(filename)
        n_atoms_check = int(xyz_check.readline())
        xyz_check.readline()  # title
        j = 0
        for line in xyz_check:
            j += 1
            if j > n_atoms_check:
                raise TypeError("Error in xyz file format. Check for empty lines.")
        xyz_check.close()

        xyz = open(filename)
        n_atoms = int(xyz.readline())
        xyz.readline()  # title

        coord_x = np.zeros(n_atoms)
        coord_y = np.zeros(n_atoms)
        coord_z = np.zeros(n_atoms)
        colores = ["black"] * n_atoms
        nombres = [None] * n_atoms

        i = 0
        for line in xyz:
            atom, x, y, z = line.split()
            coord_x[i] = x
            coord_y[i] = y
            coord_z[i] = z
            nombres[i] = atom
            i += 1
        xyz.close()

        if achiral:
            self.achiral_atoms = (coord_x, coord_y, coord_z, colores, nombres)
        else:
            self.nro_atoms = n_atoms
            self.atoms = (coord_x, coord_y, coord_z, colores, nombres)

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
        """
        atoms_name_xyz = []
        atoms_name_dirac = []
        atoms_Z = []
        for j in range(self.nro_atoms):
            atom_name_xyz = "".join(
                [i for i in str(self.atoms[4][j]) if not i.isdigit()]
            )
            atom_name_dirac = str(self.atoms[4][j])

            atoms_name_xyz.append(atom_name_xyz)
            atoms_name_dirac.append(atom_name_dirac + str(j + 1))
            atoms_Z.append(eval("mendeleev." + atom_name_xyz + ".atomic_number"))

        # We always need the nearest assymetric structure
        filename = folder + prefix_name + "_" + "0.00.xyz"
        # Remove xyz file if exist
        try:
            os.remove(filename)
        except OSError:
            pass

        if achiral is None:
            fila_0 = [self.nro_atoms, "", "", ""]
            fila_1 = ["XYZ file", "", "", ""]
            filas = np.vstack([fila_0, fila_1])
            for j in range(self.nro_atoms):
                new_line = np.array(
                    [
                        atoms_name_xyz[j],
                        self.positions[j][0],
                        self.positions[j][1],
                        self.positions[j][2] * 0,
                    ]
                )
                filas = np.vstack([filas, new_line])

            with open(filename, "ab") as f:
                np.savetxt(f, filas, fmt="%s")
        else:
            import shutil

            shutil.copyfile(achiral, folder + "/" + prefix_name + "_0.00.xyz")

        # Export the (rotated) chiral molecule
        filename = folder + prefix_name + "_" + "{:.2f}".format(z_coordinate) + ".xyz"
        fila_0 = [self.nro_atoms, "", "", ""]
        fila_1 = ["XYZ file", "", "", ""]
        filas = np.vstack([fila_0, fila_1])
        for j in range(self.nro_atoms):
            new_line = np.array(
                [
                    atoms_name_xyz[j],
                    self.positions[j][0],
                    self.positions[j][1],
                    self.positions[j][2] * z_coordinate,
                ]
            )
            filas = np.vstack([filas, new_line])

            # Remove xyz files if they exist
            try:
                os.remove(filename)
            except OSError:
                pass

            with open(filename, "w") as f:
                f.write(str(self.nro_atoms) + "\n")
                f.write("XYZ file created by pyECM\n")
                for j in range(self.nro_atoms):
                    f.write(
                        atoms_name_xyz[j]
                        + "   "
                        + "{:.6f}".format(self.positions[j][0])
                        + " "
                        + "{:.6f}".format(self.positions[j][1])
                        + " "
                        + "{:.6f}".format(self.positions[j][2] * z_coordinate)
                        + "\n"
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
            with open(filename_DIRAC, "w") as dirac:
                dirac.write("DIRAC\n")
                dirac.write("\n")
                dirac.write("\n")
                dirac.write("C " + "{:3d}".format(self.nro_atoms) + "  0 0         A\n")
                for j in range(self.nro_atoms):
                    dirac.write("     " + "{:3d}".format(atoms_Z[j]) + ".     1\n")
                    dirac.write(
                        atoms_name_dirac[j]
                        + "   "
                        + "{:.6f}".format(self.positions[j][0])
                        + " "
                        + "{:.6f}".format(self.positions[j][1])
                        + " "
                        + "{:.6f}".format(self.positions[j][2] * z_coordinate)
                        + "\n"
                    )
                    dirac.write("LARGE BASIS base\n")
                dirac.write("FINISH")

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

        :return: NORM1, CCM1, NORM2, CCM2
        :rtype: float(s)
        """

        coordenadas_x = self.positions[:, 0]
        coordenadas_y = self.positions[:, 1]
        coordenadas_z = self.positions[:, 2] * z_coordinate

        # If the achiral structure was not imported, it is assumed that
        # the molecule was rotated so that it'd defining direction
        # lies on z-axis and the molecule lies on the z = 0 plane.
        if self.XYZ_achiral_file is None:
            achiral_x_coordinates = coordenadas_x
            achiral_y_coordinates = coordenadas_y
            achiral_z_coordinates = coordenadas_z * 0
        else:
            achiral_x_coordinates = self.positions_achiral[:, 0]
            achiral_y_coordinates = self.positions_achiral[:, 1]
            achiral_z_coordinates = self.positions_achiral[:, 2]

        diff_x = coordenadas_x - achiral_x_coordinates
        diff_y = coordenadas_y - achiral_y_coordinates
        diff_z = coordenadas_z - achiral_z_coordinates
        CCM_distance = np.sum(diff_x**2 + diff_y**2 + diff_z**2)

        mean_x = np.average(coordenadas_x)
        mean_y = np.average(coordenadas_y)
        mean_z = np.average(coordenadas_z)

        # Method 1
        # https://doi.org/10.1021/ja9800941 (Eq. 1)

        mendeleev.Fe.atomic_weight
        atomic_weights = np.zeros(self.nro_atoms)
        i = 0
        for name in self.atoms[4]:
            atomic_weights[i] = eval("mendeleev." + name + ".atomic_weight")
            i = i + 1

        total_mass = np.sum(atomic_weights)
        center_of_mass_x = np.sum(coordenadas_x * atomic_weights) / total_mass
        center_of_mass_y = np.sum(coordenadas_y * atomic_weights) / total_mass
        center_of_mass_z = np.sum(coordenadas_z * atomic_weights) / total_mass
        distances_to_CM_x = coordenadas_x - center_of_mass_x
        distances_to_CM_y = coordenadas_y - center_of_mass_y
        distances_to_CM_z = coordenadas_z - center_of_mass_z
        distances_to_CM = np.sqrt(
            distances_to_CM_x**2 + distances_to_CM_y**2 + distances_to_CM_z**2
        )

        D = np.max(distances_to_CM)
        Norm_1 = D**2 * self.nro_atoms
        CCM_1 = (1 / Norm_1) * CCM_distance * 100

        # Method 2
        # https://doi.org/10.1002/chir.20807 (Eq. 1)
        Norm_2 = np.sum(
            (coordenadas_x - mean_x) ** 2
            + (coordenadas_y - mean_y) ** 2
            + (coordenadas_z - mean_z) ** 2
        )
        CCM_2 = CCM_distance / Norm_2 * 100

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
        """Obtain the wave function with the pySCF code.

        :param name: name of the xyz molecule file,
            including its directory, defaults to None
        :param cartesian: use cartesian basis set, defaults to False
        :type cartesian: bool, optional
        :param cartesian: use cartesian basis set, defaults to False
        :type cartesian: bool, optional
        :param gto_dict: Set options for Gaussian Type Orbitals, defaults to None
        :type gto_dict: dict, optional
        :param method_dict: Set method for WF calculation
            with the pySCF code, defaults to None
        :type method_dict: dict, optional
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
        else:
            NR = method_dict.get(
                "NR", True
            )  # Get 'fourcomp' from dict. False as default.
            fourcomp = method_dict.get(
                "fourcomp", False
            )  # Get 'X2C' from dict. False as default.
            X2C = method_dict.get(
                "X2C", False
            )  # Get "DFT" from dict. False as default.
            DFT = method_dict.get(
                "DFT", False
            )  # Get 'debug' from dict. Zero as default.
            debug = method_dict.get(
                "debug", 0
            )  # Get 'cvalue' from dict. 137.03599967994 as default.
            cvalue = method_dict.get("cvalue", 137.03599967994)

        if fourcomp is not False and DFT is not False:
            raise NotImplementedError("4c-DFT not available yet.")

        mol_chiral = gto.M(
            atom=name + "_" + "{:.2f}".format(z_coordinate) + ".xyz",
            max_memory=5000.0,
            **self.gto_dict
        )

        if cartesian:
            mol_chiral, ctr_coeff1 = mol_chiral.to_uncontracted_cartesian_basis()

        if NR:
            start_NRtime = time.time()
            if DFT is False:
                mf_chiral = scf.RHF(mol_chiral)
                mf_chiral.kernel()
            else:
                from pyscf import dft

                mf_chiral = dft.RKS(mol_chiral)
                mf_chiral.xc = DFT
                mf_chiral.kernel()

            from pyscf import dft

            mf_chiral = dft.RKS(mol_chiral)
            mf_chiral.xc = "pbe0"
            mf_chiral.kernel()

            overlap_chiral = mol_chiral.intor("int1e_ovlp")

            # Cantidad de orbitales atómicos
            AO_number = mol_chiral.nao

            # Cantidad de MO ocupados
            # occupied_MO = mf_chiral.mol.nelec[0]
            nelec_alpha = mf_chiral.mol.nelec[0]
            nelec_beta = mf_chiral.mol.nelec[1]
            # occupied_MO = nelec_alpha + nelec_beta

            # MO coeficients
            all_mo_coef = mf_chiral.mo_coeff
            ocupp_mo_coeff_alpha = mf_chiral.mo_coeff[0:AO_number, 0:nelec_alpha]
            ocupp_mo_coeff_beta = mf_chiral.mo_coeff[0:AO_number, 0:nelec_beta]

            norma_alpha = np.trace(
                mm(mm(tp(ocupp_mo_coeff_alpha), overlap_chiral), ocupp_mo_coeff_alpha)
            )
            norma_beta = np.trace(
                mm(mm(tp(ocupp_mo_coeff_beta), overlap_chiral), ocupp_mo_coeff_beta)
            )
            norma = norma_alpha + norma_beta

            end_NRtime = time.time()
            if debug > 0:
                print("naos_cart:", mol_chiral.nao)
                print("MO occupation", mf_chiral.mol.nelec)
                print("norma WF (original molecule):", norma)
                print("NR energy", mf_chiral.e_tot)
                print("NR time (min):", (end_NRtime - start_NRtime) / 60)
                sys.stdout.flush()

            self.NR_Noccupied_MO_alpha = nelec_alpha
            self.NR_Noccupied_MO_beta = nelec_beta
            self.NR_all_MO = all_mo_coef
            self.NR_occupied_MO = mf_chiral.mol.nelec
            self.NR_pyscf = mf_chiral

        if X2C:

            start_X2Ctime = time.time()
            if DFT is False:
                # UHF X2C (with_soc)
                mf_chiral_x2c = mol_chiral.X2C()
                mf_chiral_x2c.kernel()
            else:
                # For x2c+dft
                from pyscf.x2c import dft as x2c_dft

                mf_chiral_x2c = x2c_dft.UKS(mol_chiral)
                mf_chiral_x2c.xc = DFT
                # mf_chiral_x2c.verbose=4
                mf_chiral_x2c.kernel()

            overlap_chiral_x2c = mol_chiral.intor(
                "int1e_ovlp_spinor"
            )  # (j-adapted) spinor basis

            # Number of atomic orbitals
            AO_number = mol_chiral.nao

            # Molecular coeficients. In mf_chiral_x2c.mo_coeff orbitals are
            # ordered by: 1up, 1down, 2up, 2down, etc.
            all_mo_coef_x2c = mf_chiral_x2c.mo_coeff

            # Occupied molecular coeficients
            nelec_alpha = mf_chiral_x2c.mol.nelec[0]
            nelec_beta = mf_chiral_x2c.mol.nelec[1]

            norm_x2c = np.trace(
                mm(
                    mm(
                        tp(all_mo_coef_x2c[:, : nelec_alpha + nelec_beta]).conjugate(),
                        overlap_chiral_x2c,
                    ),
                    all_mo_coef_x2c[:, : nelec_alpha + nelec_beta],
                )
            )

            # Extracted from https://github.com/pyscf/pyscf/blob/
            #                master/examples/x2c/03-x2c_ghf.py
            # Using the j-adapted results to construct initial
            # guess for X2C-GHF, SCF can be converged to the
            # correct result in one iteration.

            # Attributes for GHF method
            # GHF orbital coefficients are 2D array.
            # Let nao be the number of spatial AOs, mo_coeff[:nao]
            # are the coefficients of AO with alpha spin; mo_coeff[nao:nao*2]
            # are the coefficients of AO with beta spin

            # I couldn't yet get the right norm in this way. JJA 2024.

            # The transformation from spin orbital basis to spinor basis
            # c = np.vstack(mol_chiral.sph2spinor_coeff())
            # Construct new initial guess from the spinor basis solution
            # mo1 = c.dot(mf_chiral_x2c.mo_coeff)
            # dm = mf_chiral_x2c.make_rdm1(mo1, mf_chiral_x2c.mo_occ)

            # x2c_ghf_mf = mol_chiral.GHF().x2c1e()
            # x2c_ghf.verbose = 4
            # x2c_ghf_mf.max_cycle = 10
            # x2c_ghf_mf.kernel(dm0=dm)

            end_X2Ctime = time.time()

            if debug > 0:
                print("X2C WF NORM (original molecule):", norm_x2c.real)
                print("X2C energy", mf_chiral_x2c.e_tot)
                print("Electrones alpha y beta:", nelec_alpha, nelec_beta)
                print("X2C time (min):", (end_X2Ctime - start_X2Ctime) / 60)
                sys.stdout.flush()

            self.x2c_MO = all_mo_coef_x2c
            self.x2c_occup_MO = all_mo_coef_x2c[:, : nelec_alpha + nelec_beta]
            self.x2c_Nalphaoccupied_MO = nelec_alpha
            self.x2c_Nbetaoccupied_MO = nelec_beta
            self.x2c_energy = mf_chiral_x2c.e_tot

        if fourcomp:
            start_4ctime = time.time()
            with light_speed(cvalue):
                c = lib.param.LIGHT_SPEED

                mf_chiral_rel = scf.DHF(mol_chiral)
                mf_chiral_rel.kernel()

                n4c, nmo = mf_chiral_rel.mo_coeff.shape
                self.n4c, self.rel_nmo = n4c, nmo
                n2c = n4c // 2
                nNeg = nmo // 2  # molecular orbitals of negative energy
                nocc = mol_chiral.nelectron
                # nvir = nmo // 2 - nocc  # virtual orbitals
                mo_pos_l = mf_chiral_rel.mo_coeff[:n2c, nNeg:]
                mo_pos_s = mf_chiral_rel.mo_coeff[n2c:, nNeg:]  # * (.5/c)
                Lo = mo_pos_l[:, :nocc]
                So = mo_pos_s[:, :nocc]
                # Lv = mo_pos_l[:,nocc:]
                # Sv = mo_pos_s[:,nocc:]

                #                from pyscf.scf.dhf import get_ovlp

                # overlap_chiral_4c = get_ovlp(mol_chiral)
                overlap_chiral_4c = get_ovlp_AUCAR(mol_chiral)

                overlap_chiral_large = overlap_chiral_4c[:n2c, :n2c]
                overlap_chiral_small = overlap_chiral_4c[n2c:, n2c:]

                AO_number = mol_chiral.nao

                # Unnused:
                overlap_chiral = mol_chiral.intor("int1e_ovlp_spinor")

                norma_LoLo_chiral = np.trace(
                    mm(mm(tp(Lo).conjugate(), overlap_chiral_large), Lo)
                )
                norma_SoSo_chiral = np.trace(
                    mm(mm(tp(So).conjugate(), overlap_chiral_small), So)
                )
                norma_total_chiral = norma_LoLo_chiral + norma_SoSo_chiral

                end_4ctime = time.time()
                if debug > 0:
                    print("LoLo Norm:", norma_LoLo_chiral)
                    print("SoSo Norm:", norma_SoSo_chiral)
                    print("Total (chiral) Norm:", norma_total_chiral)
                    print("cvalue", c)
                    print("4c energy", mf_chiral_rel.e_tot)
                    print("4c time (min):", (end_4ctime - start_4ctime) / 60)

                self.rel_MO_Lo = Lo
                self.rel_MO_So = So
                self.rel_Noccupied_MO = nocc
                self.rel_energy = mf_chiral_rel.e_tot
                self.rel_pyscf = mf_chiral_rel

        self.AO_number = AO_number

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
        :param cartesian: use cartesian basis set, defaults to False
        :type cartesian: bool, optional
        :param z_coordinate: variable for ECM on path, defaults to 1.00
        :type z_coordinate: float, optional
        :param path: True for ECM on path, defaults to False
        :type path: bool, optional
        :param method_dict: Set method for WF calculation,
            previously calculated, defaults to None
        :type method_dict: dict, optional
        :raises AttributeError: _description_
        :return: _description_
        :rtype: _type_
        """

        # Define default values for keys in method_dict
        if method_dict is None:
            method_dict = {}
        else:
            NR = method_dict.get(
                "NR", True
            )  # Get 'fourcomp' from dict. False as default.
            fourcomp = method_dict.get(
                "fourcomp", False
            )  # Get 'X2C' from dict. False as default.
            X2C = method_dict.get(
                "X2C", False
            )  # Get 'debug' from dict. Zero as default.
            debug = method_dict.get("debug", 0)
            # Get 'cvalue' from dict. 137.03599967994 as default.
            cvalue = method_dict.get("cvalue", 137.03599967994)

        ECM_NR = None
        ECM_X2C = None
        ECM_4c = None
        ECM_molcontr_alpha = []
        ECM_molcontr_beta = []
        ECM_X2C_molcontr = []

        mol_chiral = gto.M(
            atom=name + "_" + "{:.2f}".format(z_coordinate) + ".xyz",
            max_memory=5000.0,
            **self.gto_dict
        )

        mol_achiral = gto.M(atom=name + "_0.00.xyz", max_memory=5000.0, **self.gto_dict)

        if cartesian:
            mol_chiral, ctr_coeff1 = mol_chiral.to_uncontracted_cartesian_basis()
            mol_achiral, ctr_coeff2 = mol_achiral.to_uncontracted_cartesian_basis()

        mol_super = mol_chiral + mol_achiral

        if NR:
            start_NRtime = time.time()
            #            mf_chiral = scf.RHF(mol_chiral)
            #            mf_chiral.kernel()

            # from pyscf import dft
            # mf_chiral = dft.RKS(mol_chiral)
            # mf_chiral.xc = 'b3lyp'
            # mf_chiral.kernel()

            naos_sph = mol_chiral.intor("int1e_ovlp_sph").shape[0]
            Noccupied_MO_alpha = self.NR_Noccupied_MO_alpha
            Noccupied_MO_beta = self.NR_Noccupied_MO_beta

            AO_number = mol_chiral.nao
            AO_number_supermol = np.array([mol_super.nao])[0]

            overlap_mixed_fullspace = mol_super.intor("int1e_ovlp")
            overlap_mixed = overlap_mixed_fullspace[
                0 : int(AO_number_supermol / 2),
                int(AO_number_supermol / 2) : AO_number_supermol,
            ]

            overlap_chiral = mol_chiral.intor("int1e_ovlp")
            overlap_achiral = mol_achiral.intor("int1e_ovlp")

            overlap_pot_chiral = matrix_power(overlap_chiral, 0.5)
            overlap_pot_achiral = matrix_power(overlap_achiral, -0.5)

            # ocupp_mo_coeff = self.NR_all_MO[0:AO_number, 0:occupied_MO]
            ocupp_mo_coeff_alpha = self.NR_all_MO[0:AO_number, 0:Noccupied_MO_alpha]
            ocupp_mo_coeff_beta = self.NR_all_MO[0:AO_number, 0:Noccupied_MO_beta]
            # norma_chiral = np.trace(
            #    mm(mm(tp(ocupp_mo_coeff), overlap_chiral), ocupp_mo_coeff)
            # )
            norma_alpha = np.trace(
                mm(mm(tp(ocupp_mo_coeff_alpha), overlap_chiral), ocupp_mo_coeff_alpha)
            )
            norma_beta = np.trace(
                mm(mm(tp(ocupp_mo_coeff_beta), overlap_chiral), ocupp_mo_coeff_beta)
            )
            norma_chiral = norma_alpha + norma_beta

            # old basis: chiral basis set
            # new basis: achiral basis set
            C_achiral_newbasis = mm(
                mm(overlap_pot_achiral, overlap_pot_chiral), self.NR_all_MO
            )

            achiral_norm_alpha = 0
            achiral_norm_beta = 0
            solapamiento_NR_alpha = 0
            solapamiento_NR_beta = 0

            for k in range(Noccupied_MO_alpha):
                achiral_norm_alpha = (
                    achiral_norm_alpha
                    + mm(
                        mm(tp(C_achiral_newbasis).conjugate(), overlap_achiral),
                        C_achiral_newbasis,
                    )[k, k]
                )
                solapamiento_NR_alpha = (
                    solapamiento_NR_alpha
                    + mm(
                        mm(tp(self.NR_all_MO).conjugate(), overlap_mixed),
                        C_achiral_newbasis,
                    )[k, k]
                )

                ECM_molcontr_alpha.append(
                    100
                    * (
                        1
                        - np.abs(
                            mm(
                                mm(tp(self.NR_all_MO), overlap_mixed),
                                C_achiral_newbasis,
                            )[k, k]
                        )
                    )
                )

            for k in range(Noccupied_MO_beta):
                achiral_norm_beta = (
                    achiral_norm_beta
                    + mm(
                        mm(tp(C_achiral_newbasis).conjugate(), overlap_achiral),
                        C_achiral_newbasis,
                    )[k, k]
                )
                solapamiento_NR_beta = (
                    solapamiento_NR_beta
                    + mm(
                        mm(tp(self.NR_all_MO).conjugate(), overlap_mixed),
                        C_achiral_newbasis,
                    )[k, k]
                )

                ECM_molcontr_beta.append(
                    100
                    * (
                        1
                        - np.abs(
                            mm(
                                mm(tp(self.NR_all_MO), overlap_mixed),
                                C_achiral_newbasis,
                            )[k, k]
                        )
                    )
                )
            #     print("Overlap contribution from",k,"orbital:"
            # ,mm(mm(tp(mf_chiral.mo_coeff),overlap_mixed),C_achiral_newbasis)[k,k])
            #     print(" ")

            solapamiento_NR = solapamiento_NR_alpha + solapamiento_NR_beta

            achiral_norm = achiral_norm_alpha + achiral_norm_beta

            ECM_NR = 100 * (1 - np.abs(solapamiento_NR) / norma_chiral)

            ECM_NR_molcontr_alpha = np.transpose(
                np.reshape(np.ravel(ECM_molcontr_alpha), (Noccupied_MO_alpha))
            )
            ECM_NR_molcontr_alpha = ECM_NR_molcontr_alpha / norma_chiral

            ECM_NR_molcontr_beta = np.transpose(
                np.reshape(np.ravel(ECM_molcontr_beta), (Noccupied_MO_beta))
            )
            ECM_NR_molcontr_beta = ECM_NR_molcontr_beta / norma_chiral

            # Fill array with zeros for total molecular contributions
            ECM_NR_molcontr = np.zeros(
                len(ECM_NR_molcontr_alpha) + len(ECM_NR_molcontr_beta)
            )

            # Asign elements to array
            ECM_NR_molcontr[::2] = ECM_NR_molcontr_alpha
            ECM_NR_molcontr[1::2] = ECM_NR_molcontr_beta

            end_NRtime = time.time()
            if debug > 0:
                print("naos_cart:", mol_chiral.nao)
                print("naos_sph:", naos_sph)
                print("norma WF chiral (chiral basis):", norma_chiral)
                print("norma WF achiral (achiral basis):", achiral_norm)
                print("NR overlap (normalized):", solapamiento_NR / norma_chiral)
                print("NR time (min):", (end_NRtime - start_NRtime) / 60)

        if X2C:
            start_X2Ctime = time.time()

            Noccupied_MO = self.x2c_Nalphaoccupied_MO + self.x2c_Nbetaoccupied_MO
            AO_number = mol_chiral.nao
            AO_number_supermol = np.array([mol_super.nao])[0]

            overlap_mixed_fullspace = mol_super.intor("int1e_ovlp_spinor")
            overlap_mixed = overlap_mixed_fullspace[
                0 : int(AO_number_supermol),
                int(AO_number_supermol) : 2 * AO_number_supermol,
            ]

            overlap_chiral = mol_chiral.intor("int1e_ovlp_spinor")
            overlap_achiral = mol_achiral.intor("int1e_ovlp_spinor")

            overlap_pot_chiral = matrix_power(overlap_chiral, 0.5)
            overlap_pot_achiral = matrix_power(overlap_achiral, -0.5)

            norma_x2c_chiral = np.trace(
                mm(
                    mm(tp(self.x2c_occup_MO).conjugate(), overlap_chiral),
                    self.x2c_occup_MO,
                )
            )

            # old basis: chiral basis set
            # new basis: achiral basis set
            C_x2c_achiral_newbasis = mm(
                mm(overlap_pot_achiral, overlap_pot_chiral), self.x2c_MO
            )

            achiral_x2c_norm = 0
            solapamiento_x2c = 0
            for k in range(Noccupied_MO):
                achiral_x2c_norm = (
                    achiral_x2c_norm
                    + mm(
                        mm(tp(C_x2c_achiral_newbasis).conjugate(), overlap_achiral),
                        C_x2c_achiral_newbasis,
                    )[k, k]
                )

                solapamiento_x2c = (
                    solapamiento_x2c
                    + mm(
                        mm(tp(self.x2c_MO).conjugate(), overlap_mixed),
                        C_x2c_achiral_newbasis,
                    )[k, k]
                )

                ECM_X2C_molcontr.append(
                    100
                    * (
                        1
                        - np.abs(
                            mm(
                                mm(tp(self.x2c_MO), overlap_mixed),
                                C_x2c_achiral_newbasis,
                            )[k, k]
                        )
                    )
                )

            ECM_X2C = 100 * (
                1 - np.abs(solapamiento_x2c.real) / np.abs(norma_x2c_chiral)
            )

            end_X2Ctime = time.time()
            if debug > 0:
                print("X2C: norma WF chiral (chiral basis):", norma_x2c_chiral)
                print("X2C: norma WF achiral (achiral basis):", achiral_x2c_norm)
                print("X2C: overlap (normalized):", solapamiento_x2c / norma_x2c_chiral)
                print("X2C time (min):", (end_X2Ctime - start_X2Ctime) / 60)

        if fourcomp:
            from pyscf.scf.dhf import get_ovlp

            with light_speed(cvalue):

                if not hasattr(self, "rel_MO_Lo"):
                    raise AttributeError(
                        "The four-component wave function is not defined. "
                        "Try with fourcomp = False."
                    )

                n2c = self.n4c // 2
                nocc = self.rel_Noccupied_MO

                Lo = self.rel_MO_Lo  # Large MO coefficients
                So = self.rel_MO_So  # Small MO coefficients

                overlap_chiral_4c = get_ovlp(mol_chiral)
                overlap_achiral_4c = get_ovlp(mol_achiral)

                overlap_chiral_large = overlap_chiral_4c[:n2c, :n2c]
                overlap_chiral_small = overlap_chiral_4c[n2c:, n2c:]

                overlap_achiral_large = overlap_achiral_4c[:n2c, :n2c]
                overlap_achiral_small = overlap_achiral_4c[n2c:, n2c:]

                mol_super = mol_chiral + mol_achiral
                overlap_supermol_4c = get_ovlp_AUCAR(mol_super)

                # Its ok:
                # overlap_chiral_large -
                # overlap_supermol_4c[:n2c,0:n2c]
                # overlap_achiral_large -
                # overlap_supermol_4c[n2c:2*n2c,n2c:2*n2c]
                # overlap_chiral_small -
                # overlap_supermol_4c[2*n2c:3*n2c,2*n2c:3*n2c]
                # overlap_achiral_small -
                # overlap_supermol_4c[3*n2c:4*n2c,3*n2c:4*n2c]

                overlap_mixed_SchiralSachiral = overlap_supermol_4c[
                    2 * n2c : 3 * n2c, 3 * n2c : 4 * n2c
                ]
                overlap_mixed_LchiralLachiral = overlap_supermol_4c[
                    :n2c, 1 * n2c : 2 * n2c
                ]

                AO_number = mol_chiral.nao
                AO_number_supermol = np.array([mol_super.nao])[0]

                overlap_mixed_fullspace = mol_super.intor("int1e_ovlp_spinor")
                overlap_mixed = overlap_mixed_fullspace[
                    0 : int(AO_number_supermol),
                    int(AO_number_supermol) : 2 * AO_number_supermol,
                ]

                overlap_chiral = mol_chiral.intor("int1e_ovlp_spinor")
                overlap_achiral = mol_achiral.intor("int1e_ovlp_spinor")

                overlap_pot_achiral_large = matrix_power(overlap_achiral_large, -0.5)

                overlap_pot_achiral_small = matrix_power(overlap_achiral_small, -0.5)

                overlap_ll_pot_chiral = matrix_power(overlap_chiral_large, 0.5)
                overlap_ss_pot_chiral = matrix_power(overlap_chiral_small, 0.5)

                norma_LoLo_chiral = np.trace(
                    mm(mm(tp(Lo).conjugate(), overlap_chiral_large), Lo)
                )
                norma_SoSo_chiral = np.trace(
                    mm(mm(tp(So).conjugate(), overlap_chiral_small), So)
                )
                norma_total_chiral = norma_LoLo_chiral + norma_SoSo_chiral

                # MCOEFF new basis:
                # C_Lo_achiral_newbasis =
                # mm(mm(overlap_pot_achiral,overlap_ll_pot_chiral),Lo)
                # C_So_achiral_newbasis =
                # mm(mm(overlap_pot_achiral,overlap_ss_pot_chiral),So)
                C_Lo_achiral_newbasis = mm(
                    mm(overlap_pot_achiral_large, overlap_ll_pot_chiral), Lo
                )
                C_So_achiral_newbasis = mm(
                    mm(overlap_pot_achiral_small, overlap_ss_pot_chiral), So
                )

                achiral_norm_So = 0
                achiral_norm_Lo = 0
                solapamiento_LoLo = 0
                solapamiento_SoSo = 0
                ECM_4c_molcontr = []

                for k in range(nocc):
                    achiral_norm_Lo = (
                        achiral_norm_Lo
                        + mm(
                            mm(
                                tp(C_Lo_achiral_newbasis).conjugate(),
                                overlap_achiral_large,
                            ),
                            C_Lo_achiral_newbasis,
                        )[k, k]
                    )
                    achiral_norm_So = (
                        achiral_norm_So
                        + mm(
                            mm(
                                tp(C_So_achiral_newbasis).conjugate(),
                                overlap_achiral_small,
                            ),
                            C_So_achiral_newbasis,
                        )[k, k]
                    )
                    solapamiento_LoLo = (
                        solapamiento_LoLo
                        + mm(
                            mm(tp(Lo).conjugate(), overlap_mixed_LchiralLachiral),
                            C_Lo_achiral_newbasis,
                        )[k, k]
                    )
                    solapamiento_SoSo = (
                        solapamiento_SoSo
                        + mm(
                            mm(tp(So).conjugate(), overlap_mixed_SchiralSachiral),
                            C_So_achiral_newbasis,
                        )[k, k]
                    )

                    ECM_4c_molcontr.append(
                        mm(
                            mm(tp(Lo).conjugate(), overlap_mixed_LchiralLachiral),
                            C_Lo_achiral_newbasis,
                        )[k, k]
                        + mm(
                            mm(tp(So).conjugate(), overlap_mixed_SchiralSachiral),
                            C_So_achiral_newbasis,
                        )[k, k]
                    )

                solapamiento_total = (solapamiento_LoLo + solapamiento_SoSo).real
                ECM_4c_molcontr = np.array(ECM_4c_molcontr).real
                ECM_4c_molcontr = (
                    100 * (1 - ECM_4c_molcontr) / np.abs(norma_total_chiral)
                )

                ECM_4c = 100 * (
                    1 - np.abs(solapamiento_total) / np.abs(norma_total_chiral)
                )

                if debug > 0:
                    print("LoLo Norm:", norma_LoLo_chiral)
                    print("SoSo Norm:", norma_SoSo_chiral)
                    print("Total (chiral) Norm:", norma_total_chiral)
                    print("Achiral LoLo Norm:", achiral_norm_Lo)
                    print("Achiral SoSo Norm:", achiral_norm_So)
                    print("LoLo chiral/achiral overlap:", solapamiento_LoLo / nocc)
                    print("SoSo chiral/achiral overlap:", solapamiento_SoSo / nocc)
                    print("Total overlap (normalized):", solapamiento_total / nocc)
                    print("ECM LL+SS:", ECM_4c)

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
        """Calculate Gamma5 expectation value

        :param name: name of the xyz molecule file,
            including its directory, defaults to None
        :param cartesian: use cartesian basis set, defaults to False
        :type cartesian: bool, optional
        :param z_coordinate: variable for ECM on path, defaults to 1.00
        :type z_coordinate: float, optional
        :param method_dict: Set method for WF calculation,
            previously calculated, defaults to None
        :type method_dict: dict, optional
        :param debug: debugg level, defaults to 0
        :type debug: int, optional
        :raises AttributeError: check for 4c wave function
        :return: Gamma5 expectation value
        :rtype: real
        """

        if not hasattr(self, "rel_MO_Lo"):
            raise AttributeError(
                "The four-component wave function is not defined within the class. "
            )

        # Define default values for keys in method_dict
        if method_dict is None:
            method_dict = {}
        else:
            #            NR = method_dict.get(
            #                "NR", True
            #            )  # Get 'fourcomp' from dict. False as default.
            #            fourcomp = method_dict.get(
            #                "fourcomp", False
            #            )  # Get 'X2C' from dict. False as default.
            #            X2C = method_dict.get(
            #                "X2C", False
            #            )  # Get 'debug' from dict. Zero as default.
            debug = method_dict.get(
                "debug", 0
            )  # Get 'cvalue' from dict. 137.03599967994 as default.
            cvalue = method_dict.get("cvalue", 137.03599967994)

        gamma5 = 0
        # gamma5_molcontr = []

        mol_chiral = gto.M(
            atom=name + "_" + "{:.2f}".format(z_coordinate) + ".xyz",
            max_memory=5000.0,
            **self.gto_dict
        )

        if cartesian:
            mol_chiral, ctr_coeff1 = mol_chiral.to_uncontracted_cartesian_basis()

        n4c = self.n4c
        n2c = n4c // 2
        nocc = self.rel_Noccupied_MO

        Lo = self.rel_MO_Lo
        So = self.rel_MO_So

        # Taken from https://pyscf.org/_modules/pyscf/scf/dhf.html#DHF.
        s = mol_chiral.intor_symmetric("int1e_ovlp_spinor")
        t = mol_chiral.intor_symmetric("int1e_spsp_spinor")
        u = mol_chiral.intor_symmetric("int1e_sp_spinor")
        s1e = np.zeros((n4c, n4c), np.complex128)
        s1e[:n2c, :n2c] = s
        s1e[n2c:, n2c:] = t * (0.5 / cvalue) ** 2
        s1e[:n2c, n2c:] = u * (0.5 / cvalue)  # Small (ket) over large (bra)
        s1e[n2c:, :n2c] = u.conj().T * (0.5 / cvalue)  # Large (ket) over small (bra)
        # s1e is what we should get when calling to get_ovlp(mol_chiral)

        overlap_chiral_large = s1e[:n2c, :n2c]
        overlap_chiral_small = s1e[n2c:, n2c:]

        norma_LoLo_chiral = np.trace(
            mm(mm(tp(Lo).conjugate(), overlap_chiral_large), Lo)
        )
        norma_SoSo_chiral = np.trace(
            mm(mm(tp(So).conjugate(), overlap_chiral_small), So)
        )
        norma_total_chiral = norma_LoLo_chiral + norma_SoSo_chiral

        valor1 = 0
        valor2 = 0

        for k in range(nocc):
            large_on_small = mm(mm(tp(So).conjugate(), s1e[n2c:, :n2c]), Lo)[k, k]
            small_on_large = mm(mm(tp(Lo).conjugate(), s1e[:n2c, n2c:]), So)[k, k]

            valor1 = valor1 + large_on_small
            valor2 = valor2 + small_on_large

            # Molecular Contributions
            # print("mol. contr. gamma5",
            # ((large_on_small+small_on_large)*cvalue/2).real )

        gamma5 = (valor1 + valor2) / norma_total_chiral.real * cvalue / 2

        if debug > 0:
            print("LoLo Norm:", norma_LoLo_chiral)
            print("SoSo Norm:", norma_SoSo_chiral)
            print("Total (chiral) Norm:", norma_total_chiral)
            print("cvalue", cvalue)
            print("4c energy", self.rel_energy)

        return gamma5.real

    def Epv(
        self, name=None, cartesian=False, z_coordinate=1.00, method_dict=None, debug=0
    ):

        if not hasattr(self, "rel_MO_Lo"):
            raise AttributeError(
                "The four-component wave function is not defined within the class. "
            )

        self.Epv_expval = Epv_molecule(self.rel_pyscf)
