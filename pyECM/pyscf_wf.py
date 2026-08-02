"""Wave-function calculations (NR/X2C/4c) via pySCF.

Each function gets gto.Mole and returns the results within a tuple.
They dont directly read neither write attributes from the molecule class.
The molecule class is supposed to select in which of their own
attributes will save each result.
"""

import sys
import time

import numpy as np
from numpy import matmul as mm
from numpy import transpose as tp
from pyscf import lib, scf
from pyscf.lib.misc import light_speed

from pyECM.pyscf_fc import get_ovlp_AUCAR


def compute_NR_WF(mol_chiral, DFT=False, debug=0):
    """Compute the non-relativistic wave function (RHF, or RKS if a
    DFT functional is given).

    :param mol_chiral: molecule (already built) for which the WF is computed
    :type mol_chiral: pyscf.gto.Mole
    :param DFT: DFT functional name (e.g. "b3lyp"). If False, runs RHF instead
    :type DFT: str or False, optional
    :param debug: verbosity level; prints timing and norm diagnostics if > 0
    :type debug: int, optional
    :return: (Noccupied_MO_alpha, Noccupied_MO_beta, all_MO,
        occupied_MO, mf_chiral, AO_number)
    :rtype: tuple(int, int, numpy.ndarray, tuple, pyscf.scf.hf.SCF, int)
    """

    start_NRtime = time.time()
    if DFT is False:
        mf_chiral = scf.RHF(mol_chiral)
        mf_chiral.kernel()
    else:
        from pyscf import dft

        mf_chiral = dft.RKS(mol_chiral)
        mf_chiral.xc = DFT
        mf_chiral.kernel()

    overlap_chiral = mol_chiral.intor("int1e_ovlp")
    AO_number = mol_chiral.nao

    nelec_alpha = mf_chiral.mol.nelec[0]
    nelec_beta = mf_chiral.mol.nelec[1]

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

    return (
        nelec_alpha,
        nelec_beta,
        all_mo_coef,
        mf_chiral.mol.nelec,
        mf_chiral,
        AO_number,
    )


def compute_X2C_WF(mol_chiral, DFT=False, debug=0):
    """Compute the X2C (exact two-component) relativistic wave function.

    :param mol_chiral: molecule (already built) for which the WF is computed
    :type mol_chiral: pyscf.gto.Mole
    :param DFT: DFT functional name (e.g. "b3lyp"). If False, runs X2C-UHF instead
    :type DFT: str or False, optional
    :param debug: verbosity level; prints timing and norm diagnostics if > 0
    :type debug: int, optional
    :return: (x2c_MO, x2c_occup_MO, Nalphaoccupied_MO,
        Nbetaoccupied_MO, x2c_energy, AO_number)
    :rtype: tuple(numpy.ndarray, numpy.ndarray, int, int, float, int)
    """

    start_X2Ctime = time.time()
    if DFT is False:
        mf_chiral_x2c = mol_chiral.X2C()
        mf_chiral_x2c.kernel()
    else:
        from pyscf.x2c import dft as x2c_dft

        mf_chiral_x2c = x2c_dft.UKS(mol_chiral)
        mf_chiral_x2c.xc = DFT
        mf_chiral_x2c.kernel()

    overlap_chiral_x2c = mol_chiral.intor("int1e_ovlp_spinor")
    AO_number = mol_chiral.nao

    all_mo_coef_x2c = mf_chiral_x2c.mo_coeff
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

    x2c_occup_MO = all_mo_coef_x2c[:, : nelec_alpha + nelec_beta]
    return (
        all_mo_coef_x2c,
        x2c_occup_MO,
        nelec_alpha,
        nelec_beta,
        mf_chiral_x2c.e_tot,
        AO_number,
    )


def compute_4c_WF(mol_chiral, cvalue, debug=0):
    """Compute the four-component relativistic wave function (DHF).

    :param mol_chiral: molecule (already built) for which the WF is computed
    :type mol_chiral: pyscf.gto.Mole
    :param cvalue: speed of light value used for the DHF calculation
    :type cvalue: float
    :param debug: verbosity level; prints timing and norm diagnostics if > 0
    :type debug: int, optional
    :return: (n4c, rel_nmo, Lo, So, Noccupied_MO, rel_energy,
        mf_chiral_rel, AO_number)
    :rtype: tuple(int, int, numpy.ndarray, numpy.ndarray, int, float,
        pyscf.scf.dhf.DHF, int)
    """

    start_4ctime = time.time()
    with light_speed(cvalue):
        c = lib.param.LIGHT_SPEED

        mf_chiral_rel = scf.DHF(mol_chiral)
        mf_chiral_rel.kernel()

        n4c, nmo = mf_chiral_rel.mo_coeff.shape
        n2c = n4c // 2
        nNeg = nmo // 2
        nocc = mol_chiral.nelectron
        mo_pos_l = mf_chiral_rel.mo_coeff[:n2c, nNeg:]
        mo_pos_s = mf_chiral_rel.mo_coeff[n2c:, nNeg:]
        Lo = mo_pos_l[:, :nocc]
        So = mo_pos_s[:, :nocc]

        overlap_chiral_4c = get_ovlp_AUCAR(mol_chiral)
        overlap_chiral_large = overlap_chiral_4c[:n2c, :n2c]
        overlap_chiral_small = overlap_chiral_4c[n2c:, n2c:]

        AO_number = mol_chiral.nao

        LoLo_chiral_norm = np.trace(
            mm(mm(tp(Lo).conjugate(), overlap_chiral_large), Lo)
        )
        SoSo_chiral_norm = np.trace(
            mm(mm(tp(So).conjugate(), overlap_chiral_small), So)
        )
        chiral_norm = LoLo_chiral_norm + SoSo_chiral_norm

        end_4ctime = time.time()
        if debug > 0:
            print("LoLo Norm:", LoLo_chiral_norm)
            print("SoSo Norm:", SoSo_chiral_norm)
            print("Total (chiral) Norm:", chiral_norm)
            print("cvalue", c)
            print("4c energy", mf_chiral_rel.e_tot)
            print("4c time (min):", (end_4ctime - start_4ctime) / 60)

    return n4c, nmo, Lo, So, nocc, mf_chiral_rel.e_tot, mf_chiral_rel, AO_number
