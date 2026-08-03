import numpy as np
from numpy import matmul as mm
from numpy import transpose as tp


def compute_gamma5(mol_chiral, n4c, nocc, Lo, So, cvalue, rel_energy, debug=0):
    """Compute the Gamma5 expectation value for a four-component (DHF)
    wave function.

    :param mol_chiral: chiral structure the WF was computed on
    :type mol_chiral: pyscf.gto.Mole
    :param n4c: total dimension of the four-component space (2 * n2c)
    :type n4c: int
    :param nocc: number of occupied MOs
    :type nocc: int
    :param Lo: occupied MO coefficients, large component
        (obtained with pyscf_wf.compute_4c_WF)
    :type Lo: numpy.ndarray
    :param So: occupied MO coefficients, small component
        (obtained with pyscf_wf.compute_4c_WF)
    :type So: numpy.ndarray
    :param cvalue: speed of light value used in the DHF calculation
    :type cvalue: float
    :param rel_energy: total DHF energy, printed only if debug > 0
    :type rel_energy: float
    :param debug: verbosity level; prints diagnostics if > 0
    :type debug: int, optional
    :return: Gamma5 expectation value
    :rtype: float
    """
    n2c = n4c // 2

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

    LoLo_chiral_norm = np.trace(mm(mm(tp(Lo).conjugate(), overlap_chiral_large), Lo))
    SoSo_chiral_norm = np.trace(mm(mm(tp(So).conjugate(), overlap_chiral_small), So))
    chiral_norm = LoLo_chiral_norm + SoSo_chiral_norm

    term_1 = 0
    term_2 = 0

    for k in range(nocc):
        large_on_small = mm(mm(tp(So).conjugate(), s1e[n2c:, :n2c]), Lo)[k, k]
        small_on_large = mm(mm(tp(Lo).conjugate(), s1e[:n2c, n2c:]), So)[k, k]

        term_1 = term_1 + large_on_small
        term_2 = term_2 + small_on_large

    gamma5 = (term_1 + term_2) / chiral_norm.real * cvalue / 2

    if debug > 0:
        print("LoLo Norm:", LoLo_chiral_norm)
        print("SoSo Norm:", SoSo_chiral_norm)
        print("Total (chiral) Norm:", chiral_norm)
        print("cvalue", cvalue)
        print("4c energy", rel_energy)

    return gamma5.real
