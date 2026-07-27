
Installation
************

If you have pip installed:

.. code-block:: bash

    $ pip install pyECM

Or, from source code:

.. code-block:: bash

    $ git clone git@github.com:juanjoaucar/pyECM.git
    $ cd pyECM
    $ pip install -e
    
    
It's recomended to work with virtual environments. It can be done following the next few steps:

1. Install virtual environment:

.. code-block:: bash

	$ python3 -m venv venv-name

2. Activate virtual environment:

.. code-block:: bash

	$ source venv-name/bin/activate
    
3. Install the package:

.. code-block:: bash

	$ pip install pyECM

4. Run pyECM.

5. For developer only, install dependencies:

.. code-block:: bash

	$ pip install -r requirements.txt -r requirements_dev.txt

6. Run all tests:

.. code-block:: bash

	$ tox




Basic example
=============

In the following example we define the virtual mirroring path for a chiral molecule and
calculate CCM and ECM for each structure that belongs to the path.

.. code-block:: python

	from pyECM.molecule_class import molecula
	import numpy as np

	# Define the mirror path interval
	minimum = 0.50
	maximum = 0.60
	delta = 0.05
	grid_points = int (round( (maximum - minimum)/delta)) + 1

	# Define the vector that will uniquely define the virtual mirror path
	vector = np.array([-0.1807, -0.9725, -0.1469]) #Normal vector that defines the plane
	achiral_atom_origin = np.array([0.0000, 0.000, 0.0000]) # Point that defines the plane. Any atom of the symmetric structure

	# Import the molecule from a xyz file
	mymolecule = molecula(XYZ_file = '../pyECM/data/import/CFMAR_chiral.xyz', dipolo=vector, origen=achiral_atom_origin)
	mymolecule.rotate_to_align_dipole_with_z()

	# Create the files associated to the virtual mirror path
	mymolecule.xyz_mirror_path(folder='../pyECM/data/export/', prefix_name='CFMAR_chiral', lim_inf=minimum, lim_sup=maximum, points=grid_points, DIRAC = True)

	# Define the options for the ECM calculations
	options = {'cartesian' : True, 'lim_inf' : minimum, 'lim_sup' : maximum, 'points' : grid_points , 'tracking' : False, 'debug' : 0}

	# Get and save CCM and ECM values
	zrate, Norms1, CCMs1, Norms2, CCMs2 = mymolecule.CCM_on_path(lim_inf=minimum, lim_sup=maximum,points=grid_points)
	zrate, ECMs_NR, ECMs_molcontr, ECMs_4c = mymolecule.ECM_on_path(name='../pyECM/data/export/CFMAR_chiral', fourcomp=False, basis_set='sto-3g', **options)

	#Print section
	print("z rates:", zrate)
	print("NR ECMs:", ECMs_NR)


To find out what else you can do, head over to the examples under workflow folder.



====================
Development versions
====================

To install development versions of AMCESS, you can compile it from source. 
In order to install from source, you will need a python3.9 interpreter and

    * black==22.12.0
    * flake8==6.0.0
    * flake8-black==0.3.6
    * flake8-builtins==2.1.0
    * flake8-isort==6.0.0
    * tox==4.2.5
    * pytest==7.2.1
    * pytest-cov==4.0.0
    * coverage==7.0.3
    * wheel==0.38.4
    * pytest-mpl==0.16.1
    * mypy==0.991
    * sphinx==6.1.3
    * sphinx_rtd_theme==1.2.0
    * myst-parser==0.19.1
    * docutils==0.18.1
    * sphinxcontrib-bibtex==2.5.0
    * autopep8==2.0.2
    * mendeleev==0.12.1



Assuming you have already installed required dependencies, then you can compile with:

.. code-block:: bash

    $ git clone git@github.com:juanjoaucar/pyECM.git
    $ cd pyECM
    $ pip install -e


Testing
=======

.. code-block:: bash

    $ pytest --cov=pyECM/ tests --cov-append --cov-report=term-missing --cov-fail-under=90

