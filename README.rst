pcafactory
----------

The PCAFACTORY package makes easier the extraction of turbulent properties from (M)HD simulations or real observations of the interstellar medium (ISM). 
To do so, the PCAFACTORY collects features from TurbuStat, sf3dmodels, astrodendro and LIME in order to retrieve pseudo-structure functions from 
the input region using principal component analysis (and full radiative transfer techniques if necessary). 

The PCAFACTORY package is presented in the Paper II of the Cloud Factory's series. Please refer to that work if this package is useful for your research.

(Include bibtex when available).

Requirements
------------

* `sf3dmodels and LIME <https://star-forming-regions.readthedocs.io>`_
* `turbustat <https://turbustat.readthedocs.io>`_
* `astrodendro <https://dendrograms.readthedocs.io>`_

Installation
------------

You need to clone this repository. If you have a github account run on Terminal:

.. code-block:: bash

   git clone git@github.com:andizq/pcafactory.git

if you don't have one:

.. code-block:: bash

   git clone https://github.com/andizq/pcafactory.git

For simplicity in future executions include the path to the pcafactory/src files in your ~/.bashrc file (~/.profile or ~/.bash_profile for Mac users). It looks like this in my case:

.. code-block:: bash

   export PCAFACTORY="/Users/andizq/pcafactory/src"   
