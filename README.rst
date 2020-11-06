pcafactory
----------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3822718.svg
   :target: https://doi.org/10.5281/zenodo.3822718

The **pcafactory** is an analysis tool to study and extract velocity fluctuations from the interstellar medium (ISM), 
which has been proven to be a powerful proxy for understanding the nature of non-thermal motions driven by turbulent 
and/or gravitational processes taking place in the ISM. Though it may also be used for any atomic/molecular 
association with kinematical observables.

To do so, the package collects features from TurbuStat and Astrodendro to derive velocity structure functions 
by performing principal component analysis (PCA) on molecular line intensity cubes, which can be the product of 
observations or simulated data via radiative transfer. The PCA is carefully computed on selected regions of the 
input data as follows,  

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/pca_final_sketch.png?raw=true
   :align: center
 
Then, the package conveniently shows the combined spatial and spectral scales, as well as the associated velocity structure functions, 
and stores this information in a database that can be used later for further analysis. 

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/img_fit_jypxl_faceon_allportions.png?raw=true


Requirements
------------

* `turbustat <https://turbustat.readthedocs.io>`_
* `astrodendro <https://dendrograms.readthedocs.io>`_
* `sf3dmodels <https://star-forming-regions.readthedocs.io>`_ (optional, for rad. transf.)

Installation
------------

You need to clone this repository. If you have a github account run on Terminal:

.. code-block:: bash

   git clone git@github.com:andizq/pcafactory.git

if you don't have one:

.. code-block:: bash

   git clone https://github.com/andizq/pcafactory.git

For simplicity in future executions, create a variable with the path to the pcafactory/src folder in your ~/.bashrc file (~/.profile or ~/.bash_profile for Mac users). For example:

.. code-block:: bash

   export PCAFACTORY="/Users/andizq/pcafactory/src"   

Developers
----------

* `Andres Izquierdo <https://github.com/andizq>`_
* `Rowan Smith <https://www.research.manchester.ac.uk/portal/rowan.smith.html>`_

Citing the pcafactory
---------------------

The **pcafactory** is introduced in the Paper II of the Cloud Factory series. Please refer to that work if this package is useful for your research.

(Include bibtex when available).
