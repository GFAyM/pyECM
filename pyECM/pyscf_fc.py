# This should all be added to pyscf
import numpy
from pyscf import gto, lib


def integrales_fc(mf_rel, atomslist):

    c = lib.param.LIGHT_SPEED
    n4c, nmo = mf_rel.mo_coeff.shape
    n2c = n4c // 2
    fc_integrals = numpy.zeros((n4c, n4c), numpy.complex128)

    coordinates = mf_rel.mol.atom_coords()[atomslist]

    # if not four_comp:
    #    ao = gto.eval_gto(mol.mol,"GTOval_sph",coordinates)
    #    ao_fc = fac*numpy.einsum('ip,iq->pq', ao, ao)

    # Get the AO integrals in spinor form (spin index first)
    ao_spinor = gto.eval_gto(mf_rel.mol, "GTOval_spinor", coordinates, comp=1)
    ao_spinor_S = gto.eval_gto(mf_rel.mol, "GTOval_sp_spinor", coordinates, comp=1)

    # Reshape to column vectors
    ao_spinor_L_1 = ao_spinor[0, 0, :].reshape(1, -1)
    ao_spinor_L_2 = ao_spinor[1, 0, :].reshape(1, -1)

    ao_spinor_S_1 = ao_spinor_S[0, 0, :].reshape(1, -1)
    ao_spinor_S_2 = ao_spinor_S[1, 0, :].reshape(1, -1)

    # Get the matrices of the Fermi Contact integrals
    ao_spinor_matrix_LL_1 = numpy.einsum(
        "ip,iq->pq", ao_spinor_L_1.conjugate(), ao_spinor_L_1
    )
    ao_spinor_matrix_LL_2 = numpy.einsum(
        "ip,iq->pq", ao_spinor_L_2.conjugate(), ao_spinor_L_2
    )

    ao_spinor_matrix_LS_1 = numpy.einsum(
        "ip,iq->pq", ao_spinor_L_1.conjugate(), ao_spinor_S_1
    )
    ao_spinor_matrix_LS_2 = numpy.einsum(
        "ip,iq->pq", ao_spinor_L_2.conjugate(), ao_spinor_S_2
    )

    ao_spinor_matrix_SL_1 = numpy.einsum(
        "ip,iq->pq", ao_spinor_S_1.conjugate(), ao_spinor_L_1
    )
    ao_spinor_matrix_SL_2 = numpy.einsum(
        "ip,iq->pq", ao_spinor_S_2.conjugate(), ao_spinor_L_2
    )

    ao_spinor_matrix_SS_1 = numpy.einsum(
        "ip,iq->pq", ao_spinor_S_1.conjugate(), ao_spinor_S_1
    )
    ao_spinor_matrix_SS_2 = numpy.einsum(
        "ip,iq->pq", ao_spinor_S_2.conjugate(), ao_spinor_S_2
    )

    ao_spinor_matrix_LL = ao_spinor_matrix_LL_1 + ao_spinor_matrix_LL_2
    ao_spinor_matrix_LS = (ao_spinor_matrix_LS_1 + ao_spinor_matrix_LS_2) * (0.5 / c)
    ao_spinor_matrix_SL = (ao_spinor_matrix_SL_1 + ao_spinor_matrix_SL_2) * (0.5 / c)
    ao_spinor_matrix_SS = (ao_spinor_matrix_SS_1 + ao_spinor_matrix_SS_2) * (
        (0.5 / c) ** 2
    )

    fc_integrals[:n2c, :n2c] = ao_spinor_matrix_LL
    fc_integrals[n2c:, n2c:] = ao_spinor_matrix_SS
    fc_integrals[:n2c, n2c:] = ao_spinor_matrix_LS
    fc_integrals[n2c:, :n2c] = ao_spinor_matrix_SL

    return fc_integrals


def fc_expval(mf_rel, atom):
    import numpy
    from numpy import matmul as mm
    from numpy import transpose as tp

    n4c, nmo = mf_rel.mo_coeff.shape
    n2c = n4c // 2
    nNeg = nmo // 2
    nocc = mf_rel.mol.nelectron
    mo_pos_l = mf_rel.mo_coeff[:n2c, nNeg:]
    mo_pos_s = mf_rel.mo_coeff[n2c:, nNeg:]
    Lo = mo_pos_l[:, :nocc]
    So = mo_pos_s[:, :nocc]

    fac = 8 * numpy.pi / 3
    fc_ao = integrales_fc(mf_rel, [atom])
    expval_perorb = numpy.array([])
    for k in range(mf_rel.mol.nelectron):
        expval_LL_k = mm(mm(tp(Lo).conjugate(), fc_ao[:n2c, :n2c]), Lo)[k, k]
        expval_SS_k = mm(mm(tp(So).conjugate(), fc_ao[n2c:, n2c:]), Lo)[k, k]
        expval_perorb = numpy.append(expval_perorb, (expval_LL_k + expval_SS_k))

    return fac * expval_perorb.real


def Epv_atom(mf_rel, atom_index):
    import numpy
    from numpy import matmul as mm
    from numpy import transpose as tp

    n4c, nmo = mf_rel.mo_coeff.shape
    n2c = n4c // 2
    nNeg = nmo // 2  # Molecular orbitals of negative energy
    nocc = mf_rel.mol.nelectron
    mo_pos_l = mf_rel.mo_coeff[:n2c, nNeg:]
    mo_pos_s = mf_rel.mo_coeff[n2c:, nNeg:]
    Lo = mo_pos_l[:, :nocc]
    So = mo_pos_s[:, :nocc]

    # Atomic masses
    masses = mf_rel.mol.atom_mass_list(isotope_avg=False)

    # Atomic numbers (protons)
    atomic_numbers = mf_rel.mol.atom_charges()

    # Neutrons for each atom
    neutrons = []
    for mass, Z in zip(masses, atomic_numbers):
        neutron_count = mass - Z
        neutrons.append(neutron_count)

    S2THETAW = 0.23122  # AS DIRAC24 (CODATA 2018)

    # weak charge of atomic_index nucleus
    QW = (1 - 4 * S2THETAW) * atomic_numbers[atom_index] - neutrons[atom_index]

    fc_ao = integrales_fc(mf_rel, [atom_index])
    expval_perorb = numpy.array([])
    for k in range(mf_rel.mol.nelectron):
        expval_LS_k = mm(mm(tp(Lo).conjugate(), fc_ao[:n2c, n2c:]), So)[k, k]
        expval_SL_k = mm(mm(tp(So).conjugate(), fc_ao[n2c:, :n2c]), Lo)[k, k]
        expval_perorb = numpy.append(expval_perorb, (expval_LS_k + expval_SL_k))

    # fac = GF [Au] * QW * 1/(2sqrt2) *
    fac = (2.2225 * 10 ** (-14) * QW) / (2 * numpy.sqrt(2))

    return fac * expval_perorb.real


def Epv_molecule(mf_rel):
    nocc = mf_rel.mol.nelectron
    result = numpy.zeros((mf_rel.mol.natm, nocc))
    for i in range(mf_rel.mol.natm):
        result[i, :] = Epv_atom(mf_rel, i)
    return result


# To modify https://pyscf.org/_modules/pyscf/scf/dhf.html#get_ovlp
def get_ovlp_AUCAR(mol):
    n2c = mol.nao_2c()
    n4c = n2c * 2
    c = lib.param.LIGHT_SPEED

    s = mol.intor_symmetric("int1e_ovlp_spinor")
    t = mol.intor_symmetric("int1e_spsp_spinor")
    u = mol.intor_symmetric("int1e_sp_spinor")
    s1e = numpy.zeros((n4c, n4c), numpy.complex128)
    s1e[:n2c, :n2c] = s
    s1e[n2c:, n2c:] = t * (0.5 / c) ** 2
    s1e[:n2c, n2c:] = u * (0.5 / c)  # Small (ket) over large (bra)
    s1e[n2c:, :n2c] = u.conj().T * (0.5 / c)  # Large (ket) over small (bra)
    return s1e
