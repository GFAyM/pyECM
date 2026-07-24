# PyECM

## Description
The Electronic Chirality Measure (ECM) is a quantity that measures the chirality of any (chiral) molecular system. It was proposed by Luca Bellarosa and Francesco Zerbetto in 2003 [[1]](#1). In 2023 [[2]](#2), J. J. Aucar, A. Stroppa, and G. A. Aucar showed a novel, strong, and positive correlation between the energy difference of the total electronic energies of two enantiomers and ECM, supporting a subtle interplay between the weak forces acting within the nuclei of a given molecule and its chirality.

The Electronic Chirality Measure (ECM) is an end-to-end package implemented in Python 3.10 to measure the mentioned quantity. It also has some development interface with the [PySCF](https://pyscf.org/) and [DIRAC](https://www.diracprogram.org) packages.

The ecm package uses simple input files in xyz format. The so-called nearest asymmetric structure is also needed. It can be easily obtained from [this website](https://csm.ouproj.org.il/molecule). The package also allows the user to perform some simple plots to have a better understanding of the geometry of the systems under study.

The ECM is calculated computing the electronic wave function using a standard and powerful quantum chemistry package such as [PySCF](https://pyscf.org/), the Python-based Simulations of Chemistry Framework. PySCF is an efficient platform for quantum chemistry calculations that can be used to simulate the properties of molecules and crystals using mean-field and post-mean-field methods [[3]](#3).

The ecm main purpose is to be a user-friendly package, easy to install, import, and run, available on most platforms and open-source. As a Python module, ecm can be integrated into any workflow. This package has code reviews with unit testing and continuous integration, code coverage tools, and automatically keeps documentation up–to–date.

The basic [example](docs/source/quickstart.rst) demonstrates some of these features.

This package allows to:

   1. Perform basic plots in order to analyze the systems under study.
   2. Import and export molecules from/to xyz and DIRAC format.
   3. Calculate CCM and ECM in a single structure.
   4. Calculate CCM and ECM in several structures at once and in the virtual mirroring path of each one.

Read more on [ECM publications](https://pubs.acs.org/doi/pdf/10.1021/ja028646%2B).

## Requirements
First, you should install the required python packages. They can be found in the file `requirements.txt`. Developers should install `requirements_dev.txt`.

## Installation
pyECM is a **Python 3.10** package

1. Download this Git Repository

2. It is recommended to install a virtual environment:

    ```python3.10 -m venv venv_pyecm```

3. Activate the virtual environment:

    ```source venv_pyecm/bin/activate```

<!-- Install the packages through "pip install pyECM". Setup.py should be configured. -->
4. Install dependencies:

    ```pip install -r requirements.txt```

5. Run pyECM (check some examples below)

<br/><br/>

For developers,

1. Install dependencies:

      ```pip install -r requirements.txt -r requirements_dev.txt```

2. Run all test:

      ``tox``

3. To update requirements.txt and requierements_dev.txt dependencies, 

      ``pip-compile requirements.in``
      ``pip-compile requirements_dev.in``


<br/><br/>
To create local html pages,

1. Get into "docs" folder

2. Create html pages

      ```make html```

## Usage
A detailed workflow is provided in the `workflow` directory. It has a list of Jupyter notebooks with detailed examples about pyECM tools and capabilities.

1. Getting started with some basic plots.
      - `01.Plots.ipynb` 
2. Calculation of ECM in a simple molecule. 
      - `02.ECM_one-molecule.ipynb` 
3. Calculation of ECM in several molecules. 
      - `04.ECM_several-molecules.ipynb` 

## Roadmap
Ideas under consideration:

* Import and export molden files.

## Contributing
The easiest way to get help with the project is through the github project.

- GitHub:  [git@github.com:juanjoaucar/pyECM.git](https://github.com/juanjoaucar/pyECM)


## Licence
GNU General Public License v3 (GLPv3)

## Authors and Acknowledgment
Main authors: Juan Jose Aucar (_juanaucar@gmail.com_)

Advisors: Gustavo A. Aucar and Alessandro Stroppa

## Project Status
Under development

### Citation
If you find this project useful in your research, please cite:

```bibtex
@misc{2024PyECM,
  title={PyECM24 (v0.2.1)},
  author={Aucar, Juan Jose and Stroppa, Alessandro and Aucar, Gustavo Adolfo},
  howpublished={\url{https://github.com/GFAyM/pyECM}},
  year={2024},
  doi={10.5281/zenodo.21540588}
}
```

```bibtex
@article{aucar2023relationship,
  title={A relationship between the molecular parity-violation energy and the electronic chirality measure},
  author={Aucar, Juan J and Stroppa, Alessandro and Aucar, Gustavo A},
  journal={The Journal of Physical Chemistry Letters},
  volume={15},
  number={1},
  pages={234--240},
  year={2023},
  doi={10.1021/acs.jpclett.3c03038},
  publisher={ACS Publications}
}
```

And please also consider cite:

```bibtex
@article{sun2018pyscf,
  title={PySCF: the Python-based simulations of chemistry framework},
  author={Sun, Qiming and Berkelbach, Timothy C and Blunt, Nick S and Booth, George H and Guo, Sheng and Li, Zhendong and Liu, Junzi and McClain, James D and Sayfutyarova, Elvira R and Sharma, Sandeep and others},
  journal={Wiley Interdisciplinary Reviews: Computational Molecular Science},
  volume={8},
  number={1},
  pages={e1340},
  year={2018},
  doi={10.1002/wcms.1340},
  publisher={Wiley Online Library}
}
```

## Bug reports and feature requests
Please submit tickets on the [issues](https://github.com/juanjoaucar/pyECM/issues) page.

### References
<div style=font-size:12px>
      <a id="1">[1]</a> 
      Bellarosa, L., & Zerbetto, F. (2003). Enantiomeric excesses and electronic chirality measure. Journal of the American Chemical Society, 125(7), 1975-1979.
<br>
</div>
<div style=font-size:12px>
      <a id="2">[2]</a> 
      J. J. Aucar, A. Stroppa, G. A. Aucar (2033). A Relationship between the Molecular Parity-Violation Energy and the Electronic Chirality Measure. Journal of Physical Chemistry Letters, 15, 234-240.
<br>
</div>
<div style=font-size:12px>
      <a id="3">[3]</a> 
      SUN, Qiming, et al. PySCF: the Python‐based simulations of chemistry framework. Wiley Interdisciplinary Reviews: Computational Molecular Science, 2018, vol. 8, no 1, p. e1340.
<br>
</div>
