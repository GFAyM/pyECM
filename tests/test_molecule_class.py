import numpy as np
import pytest

from pyECM.molecule_class import molecula
from pyECM.data_management import obtain_nearest_structure
from numpy.testing import assert_almost_equal
import os
main_directory = os.path.realpath(os.path.dirname(__file__))+'/../'

# ===============================================================
# xyz_mirror_path function
# ===============================================================


@pytest.mark.writtenfiles
def test_create_mol_files(tmp_path):

    d = tmp_path / "sub"
    d.mkdir()
    path = str(d) + "/"

    vector = np.array([-0.1219, -0.7434, 0.6576])
    achiral_atom_origin = np.array([-1.31002, -0.33894, -0.62598])
    c = molecula(
        XYZ_file=main_directory+"pyECM/data/import/AP1_chiral.xyz",
        direction=vector,
        origen=achiral_atom_origin,
    )
    c.rotate_to_align_with_z()

#    path = str(d) + "/"

    c.xyz_mirror_path(folder=path, prefix_name="AP1", DIRAC=True,
                      lim_inf=0.5, lim_sup=1.0, points=None)
    file_050 = path + "AP1_0.50.xyz"

    text_xyz_050 = "16\nXYZ file created by pyECM\nN   -0.000005 -0.000003 -0.291276\nC   0.178754 1.090151 0.190891\nC   1.701821 1.323883 0.239674\nC   2.265610 -0.108944 0.169213\nC   1.095365 -0.939175 -0.139857\nN   2.128893 2.141943 -0.334599\nH   0.157065 0.384983 -0.754533\nH   -0.211145 0.774971 0.678523\nH   -0.373205 1.980655 0.035577\nH   1.978938 1.768910 0.722769\nH   2.560341 -0.507549 0.656199\nH   3.153188 -0.114477 -0.149593\nH   1.385702 -1.481414 -0.590544\nH   0.736826 -1.677942 0.222132\nH   3.141119 2.181181 -0.366361\nH   1.797544 3.097660 -0.288332\n"

    with open(file_050, "r") as fp:
        assert fp.read() == text_xyz_050

    text_dirac_075 = "DIRAC\n\n\nC  16  0 0         A\n       7.     1\nN1   -0.000005 -0.000003 -0.436914\nLARGE BASIS base\n       6.     1\nC2   0.178754 1.090151 0.286336\nLARGE BASIS base\n       6.     1\nC3   1.701821 1.323883 0.359511\nLARGE BASIS base\n       6.     1\nC4   2.265610 -0.108944 0.253820\nLARGE BASIS base\n       6.     1\nC5   1.095365 -0.939175 -0.209786\nLARGE BASIS base\n       7.     1\nN6   2.128893 2.141943 -0.501898\nLARGE BASIS base\n       1.     1\nH7   0.157065 0.384983 -1.131800\nLARGE BASIS base\n       1.     1\nH8   -0.211145 0.774971 1.017784\nLARGE BASIS base\n       1.     1\nH9   -0.373205 1.980655 0.053366\nLARGE BASIS base\n       1.     1\nH10   1.978938 1.768910 1.084154\nLARGE BASIS base\n       1.     1\nH11   2.560341 -0.507549 0.984299\nLARGE BASIS base\n       1.     1\nH12   3.153188 -0.114477 -0.224389\nLARGE BASIS base\n       1.     1\nH13   1.385702 -1.481414 -0.885816\nLARGE BASIS base\n       1.     1\nH14   0.736826 -1.677942 0.333199\nLARGE BASIS base\n       1.     1\nH15   3.141119 2.181181 -0.549541\nLARGE BASIS base\n       1.     1\nH16   1.797544 3.097660 -0.432498\nLARGE BASIS base\nFINISH"
    file_075_DIRAC = path + "D22_AP1_0.75.mol"
    with open(file_075_DIRAC, "r") as fp:
        assert fp.read() == text_dirac_075

    text_CCM_point_0 = str(c.CCM_on_path()[0])
    text_CCM_point_1 = str(c.CCM_on_path()[1])
    text_CCM_point_2 = str(c.CCM_on_path()[2])
    text_CCM_point_3 = str(c.CCM_on_path()[3])
    text_CCM_point_4 = str(c.CCM_on_path()[4])

    text_CCM_check_0 = "[0.2        0.28888889 0.37777778 0.46666667 0.55555556 0.64444444\n 0.73333333 0.82222222 0.91111111 1.        ]"
    text_CCM_check_1 = "[103.77109086 103.94859339 104.19064229 104.49723757 104.86837922\n 105.30406724 105.80430164 106.36908241 106.99840955 107.69228307]"
    text_CCM_check_2 = "[ 0.45649145  0.95080639  1.62215793  2.46807178  3.4854549   4.67062346\n  6.01933565  7.52682886  9.1878606  10.99675248]"
    text_CCM_check_3 = "[50.74202012 51.25666384 51.95845073 52.84738079 53.92345403 55.18667043\n 56.63703001 58.27453276 60.09917868 62.11096777]"
    text_CCM_check_4 = "[ 0.93355793  1.92823684  3.25286213  4.88021694  6.77838638  8.91221817\n 11.2447917  13.73879535 16.35773555 19.06692849]"

    assert text_CCM_point_0 == text_CCM_check_0
    assert text_CCM_point_1 == text_CCM_check_1
    assert text_CCM_point_2 == text_CCM_check_2
    assert text_CCM_point_3 == text_CCM_check_3
    assert text_CCM_point_4 == text_CCM_check_4


@pytest.mark.writtenfiles
def test_create_xyzfile(tmp_path):

    d = tmp_path / "sub"
    d.mkdir()

    vector = np.array([-0.1219, -0.7434, 0.6576])
    achiral_atom_origin = np.array([-1.31002, -0.33894, -0.62598])
    c = molecula(
        XYZ_file=main_directory+"pyECM/data/import/AP1_chiral.xyz",
        direction=vector,
        origen=achiral_atom_origin,
    )
    c.rotate_to_align_with_z()

    path = str(d) + "/"
    c.save_xyz(filename=path+"test.xyz")

    filexyz = path + "test.xyz"

#    text_xyz = '16   \nXYZ file   \nN -5.3367861987535514e-06 -2.8660119784085225e-06 -0.5825515569143256\nC 0.17875410295182356 1.0901510921606696 0.38178156508504685\nC 1.7018206179625015 1.3238830959255414 0.47934742473526704\nC 2.265610222676117 -0.10894431880701205 0.33842606756151705\nC 1.0953649015743183 -0.9391752269864793 -0.2797141224786107\nN 2.1288927712001655 2.1419427818720536 -0.6691970670272589\nH 0.1570652971982 0.3849830429626081 -1.5090668519476833\nH -0.2111448312648552 0.774970594238776 1.3570450139163897\nH -0.37320488234197635 1.9806551309842066 0.07115438357575443\nH 1.978937701216539 1.7689099268611583 1.4455380355963832\nH 2.5603405573594484 -0.5075494885888936 1.3123984352602154\nH 3.153187715485455 -0.11447739383193592 -0.2991850915621752\nH 1.3857016170021408 -1.481413846764631 -1.1810880389503868\nH 0.736826194174581 -1.6779419380690448 0.44426472755250357\nH 3.1411187824257727 2.181181032447245 -0.7327210002343334\nH 1.7975436123221973 3.097659519280735 -0.5766635350090303\n'
    text_xyz = '16\nXYZ file created by pyECM\nN   -0.000005 -0.000003 -0.582552\nC   0.178754 1.090151 0.381782\nC   1.701821 1.323883 0.479347\nC   2.265610 -0.108944 0.338426\nC   1.095365 -0.939175 -0.279714\nN   2.128893 2.141943 -0.669197\nH   0.157065 0.384983 -1.509067\nH   -0.211145 0.774971 1.357045\nH   -0.373205 1.980655 0.071154\nH   1.978938 1.768910 1.445538\nH   2.560341 -0.507549 1.312398\nH   3.153188 -0.114477 -0.299185\nH   1.385702 -1.481414 -1.181088\nH   0.736826 -1.677942 0.444265\nH   3.141119 2.181181 -0.732721\nH   1.797544 3.097660 -0.576664\n'

    with open(filexyz, "r") as fp:
        assert fp.read() == text_xyz


def test_gamma5(tmp_path, capfd):
    d = tmp_path / "sub"
    d.mkdir()
    path = str(d) + "/"


    MOL='CFMAR'
    basis='sto-3g'

    mymolecule = molecula(XYZ_file = main_directory+'pyECM/data/import/'+MOL+'_chiral.xyz', XYZ_achiral_file = main_directory+'pyECM/data/import/'+MOL+'_achiral_S20.xyz')

    mymolecule.export_xyz(folder=path, prefix_name=MOL+'_chiral', DIRAC = False, achiral=main_directory+'pyECM/data/import/'+MOL+'_achiral_S20.xyz')

    #Define options for the Gaussian Type Orbitals (generated by pySCF) and the WF calculation.
    GTO={'basis' : basis, 'charge' : 0, 'spin' : 0}
    WF_method = {'fourcomp': True, 'debug' : 1}
    
    # Get the WF with the pySCF code
    mymolecule.pySCF_WF(name=path+MOL+'_chiral', gto_dict=GTO, method_dict=WF_method)

    #Calculate gamma5
    gamma5 = mymolecule.gamma5(name=path+MOL+'_chiral', method_dict=WF_method)


    out, err = capfd.readouterr()  # For possible test on stdout
    test_result = -1.2043144522852486e-07

    assert_almost_equal(gamma5, test_result, decimal=4)

#@pytest.mark.short
def test_ecm_importing_achiral(tmp_path, capfd):
    d = tmp_path / "sub"
    d.mkdir()
    path = str(d) + "/"


    MOL='AP1'
    basis='sto-3g'

    

    mymolecule = molecula(XYZ_file = main_directory+'pyECM/data/import/'+MOL+'_chiral.xyz', XYZ_achiral_file = main_directory+'pyECM/data/import/'+MOL+'_achiral_S20.xyz')

    mymolecule.export_xyz(folder=path, prefix_name=MOL+'_chiral', DIRAC = False, achiral=main_directory+'pyECM/data/import/'+MOL+'_achiral_S20.xyz')

    # Calculate CCMs
    mymolecule.CCM()

    #Define options for the Gaussian Type Orbitals (generated by pySCF) and the WF calculation.
    GTO={'basis' : basis, 'charge' : 0, 'spin' : 0}
    WF_method = {'fourcomp': False, 'debug' : 0, 'X2C' : True}
    
    # Get the WF with the pySCF code
    mymolecule.pySCF_WF(name=path+MOL+'_chiral', gto_dict=GTO, method_dict=WF_method)

    #Calculate ECM
    method = {'fourcomp': False, 'debug' : 1, 'X2C': True}
    mymolecule.ECM(name=path+MOL+'_chiral', method_dict=method)


    out, err = capfd.readouterr()  # For possible test on stdout
    test_results = np.array([8.53405882, 14.79694024, 42.00846244, 42.01056337])

    assert_almost_equal(mymolecule.CCM1, test_results[0], decimal=4)
    assert_almost_equal(mymolecule.CCM2, test_results[1], decimal=4)
    assert_almost_equal(mymolecule.ECM_NR, test_results[2], decimal=4)
    assert_almost_equal(mymolecule.ECM_X2C, test_results[3], decimal=4)



def test_origin():
    vector = np.array([-0.1807, -0.9725, -0.1469])
    mol_A = molecula(XYZ_file=main_directory +
                     'pyECM/data/import/water.xyz', direction=vector, origen="O")
    mol_B = molecula(XYZ_file=main_directory +
                     'pyECM/data/import/water.xyz', direction=vector)


def test_ecm_onpath_cartesian(tmp_path, capfd):
    d = tmp_path / "sub"
    d.mkdir()
    path = str(d) + "/"

    minimum = 0.50
    maximum = 0.60
    delta = 0.05
    grid_points = int(round((maximum - minimum)/delta)) + 1
    vector = np.array([-0.1807, -0.9725, -0.1469])
    achiral_atom_origin = np.array([0.0000, 0.000, 0.0000])
    c = molecula(XYZ_file=main_directory+'pyECM/data/import/CFMAR_chiral.xyz',
                 direction=vector, origen=achiral_atom_origin)
    c.rotate_to_align_with_z()

    test_zrate = np.array([0.5,  0.55, 0.6])
    testECMs_NR = np.array([30.68261751, 33.2430211 , 35.58720404])
    testECMs_x2c = np.array([30.68345976, 33.24402616, 35.5884837])

    #Calculate ECM on path
    #By now we need to generate the xyz files.
    c.xyz_mirror_path(folder=path, prefix_name='CFMAR_chiral',
                      lim_inf=minimum, lim_sup=maximum, points=grid_points, DIRAC=True)

    #Define options for the Gaussian Type Orbitals (generated by pySCF) and the WF calculation.
    GTO={'basis' : 'sto-3g', 'charge' : 0, 'spin' : 0}
    method = {'fourcomp': False, 'debug' : 1, 'X2C': True}
    zrate, ECMs_NR, ECMs_x2c, ECMs_4c = c.ECM_on_path(name=path+'CFMAR_chiral', method_dict=method, gto_dict=GTO, lim_inf=minimum, lim_sup=maximum, points=grid_points, cartesian=False)
    

    out, err = capfd.readouterr()  # For possible test on stdout


    assert_almost_equal(zrate, test_zrate, decimal=4)
    assert_almost_equal(ECMs_NR, testECMs_NR, decimal=4)
    assert_almost_equal(ECMs_x2c, testECMs_x2c, decimal=4)


@pytest.mark.long
def test_ecm_onpath_4c(tmp_path, capfd):
    d = tmp_path / "sub"
    d.mkdir()
    path = str(d) + "/"

    minimum = 0.50
    maximum = 0.60
    delta = 0.05
    grid_points = int(round((maximum - minimum)/delta)) + 1
    vector = np.array([-0.1807, -0.9725, -0.1469])
    achiral_atom_origin = np.array([0.0000, 0.000, 0.0000])
    c = molecula(XYZ_file=main_directory+'pyECM/data/import/CFMAR_chiral.xyz',
                 direction=vector, origen=achiral_atom_origin)
    c.rotate_to_align_with_z()

    test_zrate = np.array([0.5,  0.55, 0.6])
    testECMs_NR = np.array([30.68260226, 33.24298947, 35.58723875])
    testECMs_4c = np.array([30.68815322, 33.24818727, 35.59219416])

    #Calculate ECM on path
    #By now we need to generate the xyz files.
    c.xyz_mirror_path(folder=path, prefix_name='CFMAR_chiral',
                      lim_inf=minimum, lim_sup=maximum, points=grid_points, DIRAC=True)
    

    #Define options for the Gaussian Type Orbitals (generated by pySCF) and the WF calculation.
    GTO={'basis' : 'sto-3g', 'charge' : 0, 'spin' : 0}
    

    method = {'fourcomp': True, 'debug' : 1}
    zrate, ECMs_NR, ECMs_x2c, ECMs_4c = c.ECM_on_path(name=path+'CFMAR_chiral', method_dict=method, gto_dict=GTO, lim_inf=minimum, lim_sup=maximum, points=grid_points)


    out, err = capfd.readouterr()  # For possible test on stdout

    assert_almost_equal(zrate, test_zrate, decimal=4)
    assert_almost_equal(ECMs_NR, testECMs_NR, decimal=4)
    assert_almost_equal(ECMs_4c, testECMs_4c, decimal=4)


#@pytest.mark.long
def test_ecm_and_epv_molecular_contributions(tmp_path, capfd):
    d = tmp_path / "sub"
    d.mkdir()
    path = str(d) + "/"


    MOL='CFMAR'
    basis='sto6g'

    

    mymolecule = molecula(XYZ_file = main_directory+'pyECM/data/import/'+MOL+'_chiral.xyz', XYZ_achiral_file = main_directory+'pyECM/data/import/'+MOL+'_achiral_S20.xyz')
    mymolecule.export_xyz(folder=path, prefix_name=MOL+'_chiral', DIRAC = False, achiral=main_directory+'pyECM/data/import/'+MOL+'_achiral_S20.xyz')

    # Calculate CCMs
    mymolecule.CCM()

    #Define options for the Gaussian Type Orbitals (generated by pySCF) and the WF calculation.
    GTO={'basis' : basis, 'charge' : 0, 'spin' : 0}
    WF_method = {'fourcomp': True, 'debug' : 0, 'X2C' : False}
    
    # Get the WF with the pySCF code
    mymolecule.pySCF_WF(name=path+MOL+'_chiral', gto_dict=GTO, method_dict=WF_method)

    #Calculate ECM
    method = {'fourcomp': True, 'debug' : 0, 'X2C': False}
    mymolecule.ECM(name=path+MOL+'_chiral', method_dict=method)

    #Calculate Epv
    mymolecule.Epv()


    out, err = capfd.readouterr()  # For possible test on stdout
    test_results_total = np.array([5.686860, 16.168746, 46.204955, 46.210173])

    test_ECM_molcontr_4c = np.array([2.38085955, 2.38085955, 2.24588134, 2.24588134, 0.22427534, 0.22427534,
 0.72965621, 0.72965621, 2.28096487, 2.28096487, 2.39839725, 2.39839725,
 2.39544703, 2.39544703, 2.40293749, 2.40293749, 0.3974209,  0.3974209,
 0.08154737, 0.08154737, 0.53669565, 0.53669565, 0.45856955, 0.45856955,
 0.32715822, 0.32715822, 0.22108933, 0.22108933, 0.38637887, 0.38637887,
 0.76293019, 0.76293019, 0.94364121, 0.94364121, 1.05984176, 1.05984176,
 1.13320732, 1.13320732, 1.5384561,  1.5384561,  0.19973077, 0.19973077] )
    
    test_ECM_molcontr_NR = np.array([2.38092297, 2.38092297, 2.24528026, 2.24528026, 0.22396535, 0.22396535,
 0.72928197, 0.72928197, 2.27653335, 2.27653335, 2.37245639, 2.37245639,
 2.31498785, 2.31498785, 2.35986081, 2.35986081, 0.39694388, 0.39694388,
 0.08042556, 0.08042556, 0.53153842, 0.53153842, 0.46753853, 0.46753853,
 0.32471434, 0.32471434, 0.22208567, 0.22208567, 0.38663981, 0.38663981,
 0.76437183, 0.76437183, 0.94454778, 0.94454778, 1.05901838, 1.05901838,
 1.00428576, 1.00428576, 1.66399638, 1.66399638, 0.2041611,  0.2041611])
    
    test_Epv_molcontr_Epv_sumovernuclei = np.array([7.27912738e-21, -7.66501226e-20,  4.77073817e-20, -1.05177251e-20])
    

    assert_almost_equal(mymolecule.CCM1, test_results_total[0], decimal=4)
    assert_almost_equal(mymolecule.CCM2, test_results_total[1], decimal=4)
    assert_almost_equal(mymolecule.ECM_NR, test_results_total[2], decimal=4)
    assert_almost_equal(mymolecule.ECM_4c, test_results_total[3], decimal=4)

    assert_almost_equal(mymolecule.ECM_NR_molcontr, test_ECM_molcontr_NR, decimal=4)
    assert_almost_equal(mymolecule.ECM_4c_molcontr, test_ECM_molcontr_4c, decimal=4)
    assert_almost_equal(np.sum(mymolecule.Epv_expval, axis=1)[:4], test_Epv_molcontr_Epv_sumovernuclei, decimal=4)


#@pytest.mark.short
def test_website_interface(tmp_path, capfd):
    d = tmp_path / "sub"
    d.mkdir()
    path = str(d) + "/"

    MOL='AP1'
    basis='sto-3g'



    obtain_nearest_structure(MOL,os.path.abspath(main_directory+'pyECM/data/import/'),output_path=main_directory+'pyECM/data/export/')

    mymolecule = molecula(XYZ_file = main_directory+'pyECM/data/import/'+MOL+'.xyz', XYZ_achiral_file = main_directory+'pyECM/data/import/'+MOL+'_achiral_S20.xyz')
    mymolecule.export_xyz(folder=path, prefix_name=MOL+'_chiral', DIRAC = False, achiral=main_directory+'pyECM/data/import/'+MOL+'_achiral_S20.xyz')

    # Calculate CCMs
    mymolecule.CCM()

    #Define options for the Gaussian Type Orbitals (generated by pySCF) and the WF calculation.
    GTO={'basis' : basis, 'charge' : 0, 'spin' : 0}
    WF_method = {'fourcomp': False, 'debug' : 0, 'X2C' : True}
    
    # Get the WF with the pySCF code
    mymolecule.pySCF_WF(name=path+MOL+'_chiral', gto_dict=GTO, method_dict=WF_method)

    #Calculate ECM
    method = {'fourcomp': False, 'debug' : 1, 'X2C': True}
    mymolecule.ECM(name=path+MOL+'_chiral', method_dict=method)

    out, err = capfd.readouterr()  # For possible test on stdout
    test_results = np.array([8.53405882, 14.79694024, 42.00846244, 42.01056337])

    assert_almost_equal(mymolecule.CCM1, test_results[0], decimal=4)
    assert_almost_equal(mymolecule.CCM2, test_results[1], decimal=4)
    assert_almost_equal(mymolecule.ECM_NR, test_results[2], decimal=4)
    assert_almost_equal(mymolecule.ECM_X2C, test_results[3], decimal=4)



import pytest
from unittest.mock import patch, MagicMock
from selenium.common.exceptions import WebDriverException
import os
from pyECM.data_management import download_CSM_zip

@patch("pyECM.data_management.webdriver.Chrome")
@patch("pyECM.data_management.webdriver.Firefox")
@patch("pyECM.data_management.webdriver.Edge")
@patch("pyECM.data_management.webdriver.Safari")
def test_download_CSM_zip_no_supported_browser(chrome_mock, firefox_mock, edge_mock, safari_mock, tmp_path):
    # Simulate that all browser initializations fail with WebDriverException
    chrome_mock.side_effect = WebDriverException
    firefox_mock.side_effect = WebDriverException
    edge_mock.side_effect = WebDriverException
    safari_mock.side_effect = WebDriverException

    input_path = str(tmp_path)
    xyz_FILENAME = "test_molecule"
    download_dir = str(tmp_path / "downloads")

    # Create the download directory
    os.makedirs(download_dir, exist_ok=True)

    with pytest.raises(SystemExit) as e:
        download_CSM_zip(input_path, xyz_FILENAME, download_dir, silent=True)

    # Verify that the function exits with code 1
    assert e.value.code == 1