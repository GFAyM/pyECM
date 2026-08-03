"""ECM (Electronic Chirality Measure) calculations via pySCF.

Each function receive a gto.Mole object and the results corresponding
to a specific wave-function (previously obtained through pySCF)
and returns ECM (including orbital contributions).
It does not read neither write self attributes.
"""

import numpy as np
from numpy import matmul as mm
from numpy import transpose as tp
from pyscf.lib.misc import light_speed
from scipy.linalg import fractional_matrix_power as matrix_power

from pyECM.decorators import debug_timed
from pyECM.pyscf_fc import get_ovlp_AUCAR


def _orbital_overlaps_and_contributions(
    n_occupied, mo_coeff, overlap_mixed, overlap_achiral, achiral_mo_coeff
):
    """Accumulate, orbital by orbital, the achiral norm, the chiral/achiral
    overlap, and each orbital's (unnormalized) ECM contribution.

    Shared by compute_ECM_NR (called once per spin channel) and
    compute_ECM_X2C (called once, over all occupied MOs).

    :param n_occupied: number of occupied MOs to sum over
    :type n_occupied: int
    :param mo_coeff: chiral MO coefficients
    :type mo_coeff: numpy.ndarray
    :param overlap_mixed: cross overlap block between the chiral and
        achiral structures (from the supermolecule)
    :type overlap_mixed: numpy.ndarray
    :param overlap_achiral: AO overlap matrix of the achiral structure
    :type overlap_achiral: numpy.ndarray
    :param achiral_mo_coeff: chiral MO coefficients already projected into
        the achiral basis (see _achiral_basis_projection)
    :type achiral_mo_coeff: numpy.ndarray
    :return: (achiral_norm, overlap_sum, molcontr), where molcontr is a
        list with each orbital's ECM contribution (not yet normalized
        by the chiral norm)
    :rtype: tuple(complex, complex, list)
    """
    achiral_norm = 0
    overlap_sum = 0
    molcontr = []
    for k in range(n_occupied):
        achiral_norm = (
            achiral_norm
            + mm(
                mm(tp(achiral_mo_coeff).conjugate(), overlap_achiral), achiral_mo_coeff
            )[k, k]
        )
        overlap_sum = (
            overlap_sum
            + mm(mm(tp(mo_coeff).conjugate(), overlap_mixed), achiral_mo_coeff)[k, k]
        )
        molcontr.append(
            100
            * (1 - np.abs(mm(mm(tp(mo_coeff), overlap_mixed), achiral_mo_coeff)[k, k]))
        )
    return achiral_norm, overlap_sum, molcontr


@debug_timed("NR")
def compute_ECM_NR(
    mol_chiral,
    mol_achiral,
    mol_super,
    NR_all_MO,
    Noccupied_MO_alpha,
    Noccupied_MO_beta,
    debug=0,
):
    """Compute ECM at non-relativistic level. It inner projects the wave-function
    from the molecule over its nearest achiral symmetric structure.

    :param mol_chiral: chiral structure
    :type mol_chiral: pyscf.gto.Mole
    :param mol_achiral: nearest achiral symmetric structure
    :type mol_achiral: pyscf.gto.Mole
    :param mol_super: molecule composed by the sum quiral+achiral, for overlap integrals
    :type mol_super: pyscf.gto.Mole
    :param NR_all_MO: molecular coeficients of the NR-WF for the chiral molecule
        (obtained through pyscf_wf.compute_NR_WF)
    :type NR_all_MO: numpy.ndarray
    :param Noccupied_MO_alpha: number of molecular orbitals with alpha spin
    :type Noccupied_MO_alpha: int
    :param Noccupied_MO_beta: number of molecular orbitals with beta spin
    :type Noccupied_MO_beta: int
    :param debug: verbosity level; prints diagnostics if > 0
    :type debug: int, optional
    :return: (ECM_NR, ECM_NR_molcontr_alpha, ECM_NR_molcontr_beta, ECM_NR_molcontr)
    :rtype: tuple(float, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """

    naos_sph = mol_chiral.intor("int1e_ovlp_sph").shape[0]

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

    ocupp_mo_coeff_alpha = NR_all_MO[0:AO_number, 0:Noccupied_MO_alpha]
    ocupp_mo_coeff_beta = NR_all_MO[0:AO_number, 0:Noccupied_MO_beta]
    norma_alpha = np.trace(
        mm(mm(tp(ocupp_mo_coeff_alpha), overlap_chiral), ocupp_mo_coeff_alpha)
    )
    norma_beta = np.trace(
        mm(mm(tp(ocupp_mo_coeff_beta), overlap_chiral), ocupp_mo_coeff_beta)
    )
    norma_chiral = norma_alpha + norma_beta

    C_achiral_newbasis = mm(mm(overlap_pot_achiral, overlap_pot_chiral), NR_all_MO)

    achiral_norm_alpha = 0
    achiral_norm_beta = 0
    overlap_NR_alpha = 0
    overlap_NR_beta = 0
    ECM_molcontr_alpha = []
    ECM_molcontr_beta = []

    achiral_norm_alpha, overlap_NR_alpha, ECM_molcontr_alpha = (
        _orbital_overlaps_and_contributions(
            Noccupied_MO_alpha,
            NR_all_MO,
            overlap_mixed,
            overlap_achiral,
            C_achiral_newbasis,
        )
    )
    achiral_norm_beta, overlap_NR_beta, ECM_molcontr_beta = (
        _orbital_overlaps_and_contributions(
            Noccupied_MO_beta,
            NR_all_MO,
            overlap_mixed,
            overlap_achiral,
            C_achiral_newbasis,
        )
    )

    overlap_NR = overlap_NR_alpha + overlap_NR_beta
    achiral_norm = achiral_norm_alpha + achiral_norm_beta

    ECM_NR = 100 * (1 - np.abs(overlap_NR) / norma_chiral)

    ECM_NR_molcontr_alpha = np.transpose(
        np.reshape(np.ravel(ECM_molcontr_alpha), (Noccupied_MO_alpha))
    )
    ECM_NR_molcontr_alpha = ECM_NR_molcontr_alpha / norma_chiral

    ECM_NR_molcontr_beta = np.transpose(
        np.reshape(np.ravel(ECM_molcontr_beta), (Noccupied_MO_beta))
    )
    ECM_NR_molcontr_beta = ECM_NR_molcontr_beta / norma_chiral

    ECM_NR_molcontr = np.zeros(len(ECM_NR_molcontr_alpha) + len(ECM_NR_molcontr_beta))
    ECM_NR_molcontr[::2] = ECM_NR_molcontr_alpha
    ECM_NR_molcontr[1::2] = ECM_NR_molcontr_beta

    if debug > 0:
        print("naos_cart:", mol_chiral.nao)
        print("naos_sph:", naos_sph)
        print("norma WF chiral (chiral basis):", norma_chiral)
        print("norma WF achiral (achiral basis):", achiral_norm)
        print("NR overlap (normalized):", overlap_NR / norma_chiral)

    return ECM_NR, ECM_NR_molcontr_alpha, ECM_NR_molcontr_beta, ECM_NR_molcontr


@debug_timed("X2C")
def compute_ECM_X2C(
    mol_chiral,
    mol_achiral,
    mol_super,
    x2c_MO,
    x2c_occup_MO,
    Nalphaoccupied_MO,
    Nbetaoccupied_MO,
    debug=0,
):
    """Compute ECM at X2C level. It inner projects the wave-function
    from the molecule over its nearest achiral symmetric structure.

    :param mol_chiral: chiral structure
    :type mol_chiral: pyscf.gto.Mole
    :param mol_achiral: nearest symmetric (achiral) reference structure
    :type mol_achiral: pyscf.gto.Mole
    :param mol_super: molecule composed by the sum quiral+achiral, for overlap integrals
    :type mol_super: pyscf.gto.Mole
    :param x2c_MO: coefficients of all MOs of the X2C WF of mol_chiral
        (obtained with pyscf_wf.compute_X2C_WF)
    :type x2c_MO: numpy.ndarray
    :param x2c_occup_MO: coefficients of the occupied MOs of the X2C WF
    :type x2c_occup_MO: numpy.ndarray
    :param Nalphaoccupied_MO: number of occupied alpha-spin MOs
    :type Nalphaoccupied_MO: int
    :param Nbetaoccupied_MO: number of occupied beta-spin MOs
    :type Nbetaoccupied_MO: int
    :param debug: verbosity level; prints diagnostics if > 0
    :type debug: int, optional
    :return: (ECM_X2C, ECM_X2C_molcontr)
    :rtype: tuple(float, list)

    """

    Noccupied_MO = Nalphaoccupied_MO + Nbetaoccupied_MO
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
        mm(mm(tp(x2c_occup_MO).conjugate(), overlap_chiral), x2c_occup_MO)
    )

    C_x2c_achiral_newbasis = mm(mm(overlap_pot_achiral, overlap_pot_chiral), x2c_MO)

    achiral_x2c_norm = 0
    solapamiento_x2c = 0
    ECM_X2C_molcontr = []

    achiral_x2c_norm, solapamiento_x2c, ECM_X2C_molcontr = (
        _orbital_overlaps_and_contributions(
            Noccupied_MO,
            x2c_MO,
            overlap_mixed,
            overlap_achiral,
            C_x2c_achiral_newbasis,
        )
    )

    ECM_X2C = 100 * (1 - np.abs(solapamiento_x2c.real) / np.abs(norma_x2c_chiral))

    if debug > 0:
        print("X2C: norma WF chiral (chiral basis):", norma_x2c_chiral)
        print("X2C: norma WF achiral (achiral basis):", achiral_x2c_norm)
        print("X2C: overlap (normalized):", solapamiento_x2c / norma_x2c_chiral)

    return ECM_X2C, ECM_X2C_molcontr


@debug_timed("4c")
def compute_ECM_4c(mol_chiral, mol_achiral, n4c, Lo, So, nocc, cvalue, debug=0):
    """Compute ECM at four-component level. It inner projects the wave-function
    from the molecule over its nearest achiral symmetric structure.

    :param mol_chiral: chiral structure
    :type mol_chiral: pyscf.gto.Mole
    :param mol_achiral: nearest symmetric (achiral) reference structure
    :type mol_achiral: pyscf.gto.Mole
    :param n4c: total dimension of the four-component space (2 * n2c)
    :type n4c: int
    :param Lo: occupied MO coefficients, large component
        (obtained with pyscf_wf.compute_4c_WF)
    :type Lo: numpy.ndarray
    :param So: occupied MO coefficients, small component
        (obtained with pyscf_wf.compute_4c_WF)
    :type So: numpy.ndarray
    :param nocc: number of occupied MOs
    :type nocc: int
    :param cvalue: speed of light value used in the DHF calculation
    :type cvalue: float
    :param debug: verbosity level; prints diagnostics if > 0
    :type debug: int, optional
    :return: (ECM_4c, ECM_4c_molcontr)
    :rtype: tuple(float, numpy.ndarray)
    """
    from pyscf.scf.dhf import get_ovlp

    with light_speed(cvalue):
        n2c = n4c // 2

        overlap_chiral_4c = get_ovlp(mol_chiral)
        overlap_achiral_4c = get_ovlp(mol_achiral)

        overlap_chiral_large = overlap_chiral_4c[:n2c, :n2c]
        overlap_chiral_small = overlap_chiral_4c[n2c:, n2c:]
        overlap_achiral_large = overlap_achiral_4c[:n2c, :n2c]
        overlap_achiral_small = overlap_achiral_4c[n2c:, n2c:]

        mol_super = mol_chiral + mol_achiral
        overlap_supermol_4c = get_ovlp_AUCAR(mol_super)

        overlap_mixed_SchiralSachiral = overlap_supermol_4c[
            2 * n2c : 3 * n2c, 3 * n2c : 4 * n2c
        ]
        overlap_mixed_LchiralLachiral = overlap_supermol_4c[:n2c, 1 * n2c : 2 * n2c]

        overlap_pot_achiral_large = matrix_power(overlap_achiral_large, -0.5)
        overlap_pot_achiral_small = matrix_power(overlap_achiral_small, -0.5)
        overlap_ll_pot_chiral = matrix_power(overlap_chiral_large, 0.5)
        overlap_ss_pot_chiral = matrix_power(overlap_chiral_small, 0.5)

        LoLo_chiral_norm = np.trace(
            mm(mm(tp(Lo).conjugate(), overlap_chiral_large), Lo)
        )
        SoSo_chiral_norm = np.trace(
            mm(mm(tp(So).conjugate(), overlap_chiral_small), So)
        )
        chiral_norm = LoLo_chiral_norm + SoSo_chiral_norm

        C_Lo_achiral_newbasis = mm(
            mm(overlap_pot_achiral_large, overlap_ll_pot_chiral), Lo
        )
        C_So_achiral_newbasis = mm(
            mm(overlap_pot_achiral_small, overlap_ss_pot_chiral), So
        )

        achiral_norm_So = 0
        achiral_norm_Lo = 0
        overlap_LoLo = 0
        overlap_SoSo = 0
        ECM_4c_molcontr = []

        for k in range(nocc):
            achiral_norm_Lo = (
                achiral_norm_Lo
                + mm(
                    mm(tp(C_Lo_achiral_newbasis).conjugate(), overlap_achiral_large),
                    C_Lo_achiral_newbasis,
                )[k, k]
            )
            achiral_norm_So = (
                achiral_norm_So
                + mm(
                    mm(tp(C_So_achiral_newbasis).conjugate(), overlap_achiral_small),
                    C_So_achiral_newbasis,
                )[k, k]
            )
            overlap_LoLo = (
                overlap_LoLo
                + mm(
                    mm(tp(Lo).conjugate(), overlap_mixed_LchiralLachiral),
                    C_Lo_achiral_newbasis,
                )[k, k]
            )
            overlap_SoSo = (
                overlap_SoSo
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

        solapamiento_total = (overlap_LoLo + overlap_SoSo).real
        ECM_4c_molcontr = np.array(ECM_4c_molcontr).real
        ECM_4c_molcontr = 100 * (1 - ECM_4c_molcontr) / np.abs(chiral_norm)

        ECM_4c = 100 * (1 - np.abs(solapamiento_total) / np.abs(chiral_norm))

        if debug > 0:
            print("LoLo Norm:", LoLo_chiral_norm)
            print("SoSo Norm:", SoSo_chiral_norm)
            print("Total (chiral) Norm:", chiral_norm)
            print("Achiral LoLo Norm:", achiral_norm_Lo)
            print("Achiral SoSo Norm:", achiral_norm_So)
            print("LoLo chiral/achiral overlap:", overlap_LoLo / nocc)
            print("SoSo chiral/achiral overlap:", overlap_SoSo / nocc)
            print("Total overlap (normalized):", solapamiento_total / nocc)
            print("ECM LL+SS:", ECM_4c)

    return ECM_4c, ECM_4c_molcontr
