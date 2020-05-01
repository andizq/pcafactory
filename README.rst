pcafactory
----------

The **pcafactory** facilitates the extraction of non-thermal kinematical signatures from the cold interstellar medium (ISM) via molecular line emission.
To do so, the **pcafactory** collects features from TurbuStat, sf3dmodels and astrodendro in order to retrieve velocity pseudo-structure functions using principal component analysis (and full radiative transfer with LIME for gas simulations). 

The PCAFACTORY package is presented in the Paper II of the Cloud Factory series. Please refer to that work if this package is useful for your research.

(Include bibtex when available).

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/img_fit_jypxl_faceon_allportions.png?raw=true

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

For simplicity in future executions, create a variable with the path to the pcafactory/src files in your ~/.bashrc file (~/.profile or ~/.bash_profile for Mac users). For example:

.. code-block:: bash

   export PCAFACTORY="/Users/andizq/pcafactory/src"   
