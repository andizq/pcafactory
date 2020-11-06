pcafactory
----------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3822718.svg
   :target: https://doi.org/10.5281/zenodo.3822718

The **pcafactory** is an analysis tool to study and extract velocity fluctuations from the interstellar medium (ISM). 
This has been proven to be a powerful proxy to understanding the nature of non-thermal motions driven by turbulent 
and/or gravitational processes taking place in the ISM. Though it may also be used for any atomic/molecular 
association with kinematical observables.

To do so, the package collects features from TurbuStat and astrodendro to derive velocity structure functions 
by performing principal component analysis (PCA) on molecular line intensity cubes, which can be the product of 
observations or simulated data via radiative transfer. The PCA is carefully computed on selected regions of the 
input data as illustrated below,  

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/pca_final_sketch.png?raw=true

Then, the package shows the resulting spatial and spectral scales, as well as the associated velocity structure functions, 
and stores this information in a database that can be used later for further analysis.

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/img_fit_jypxl_faceon_allportions.png?raw=true

.. image:: https://render.githubusercontent.com/view/pdf?commit=dea0ca52653efd5d2b83351e08b5a0e41e9b48a1&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f616e64697a712f616e64697a712e6769746875622e696f2f646561306361353236353365666435643262383333353165303862356130653431653962343861312f706361666163746f72792d646174612f6578616d706c65732d646174612f636c64425f636c6f7564666163746f72792f5043415f6a7970786c5f666163656f6e5f6f6666736574735f76657273696f6e312e706466&nwo=andizq%2Fandizq.github.io&path=pcafactory-data%2Fexamples-data%2FcldB_cloudfactory%2FPCA_jypxl_faceon_offsets_version1.pdf&repository_id=164734226&repository_type=Repository#3af4cf48-057f-40c1-a108-f720f87632d6


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
