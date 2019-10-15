(Under development)

Cloud Complex description
-------------------------

* For this example we use a self-gravitating cloud complex inmersed in the galactic potential of a Milky Way-type galaxy. 
* We let the object evolve for 2 Myr under local gravitational forces in a high resolution portion of the simulation mesh. 
* There are random supernova explosions located all across the density distribution of the galaxy.
* There is also chemical evolution for CO and Hydrogen species; sink particle formation (stellar systems); radiative heating and cooling and galactic differential rotation.

For full details on the simulation setup of this (cloud complex B) and other types of cloud complexes see the papers I and II of the Cloud Factory's series: Smith et al. subm. and Izquierdo et al. in prep (https://github.com/andizq/andizq.github.io/tree/master/pcafactory-data). 

Our simulations are powered by a customised version of the AREPO code (Springel+2010)

pcafactory-data repository
--------------------------

The data required for this example can be downloaded from `here <https://girder.hub.yt/#user/5da06b5868085e00016c2dee/folder/5da06ef668085e00016c2df3>`_.

There you will find the following files:
 
* Simulation snapshot of the cloud complex.
* 12CO J=1-0 intensity cubes: 3 line intensity cubes for different cloud orientations (face-on, edge-on phi=0, edge-on phi=90) generated with sf3dmodels and LIME (https://github.com/andizq/star-forming-regions).
* Optical depth cube for the edge-on phi=90 case.

Quick Tutorial
--------------

1. Download the simulation snapshot 
   
.. code-block:: bash

   curl https://girder.hub.yt/api/v1/item/5da0777768085e00016c2e01/download -o Restest_extracted_001_240

2. Read the snapshot and save the formatted physical information of the cloud for radiative transfer calculations.

.. code-block:: bash
      
   python make_arepo_lime.py

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/cellsize_numdens-AREPOgrid.png?raw=true
   :width: 49.5%

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/3Dpoints_snap.png?raw=true
   :width: 49.5%


3. The output files are stored by default in the folder ./Subgrids

.. code-block:: bash
   
   cd Subgrids

4. Download the CO excitation information from the LAMDA database. 

.. code-block:: bash
   
   curl https://home.strw.leidenuniv.nl/~moldata/datafiles/co.dat -o co.dat 

5. We customised the LIME code to model the radiative transfer of Arepo-like (non-uniform) meshes. It is freely available `here <https://github.com/andizq/star-forming-regions>`_. The flag -S indicates that the grid was created/processed using `sf3dmodels <https://github.com/andizq/star-forming-regions>`_, and the flag -G is for non-uniform grids. The flag -n is to show log messages on the current terminal. We call 8 cores by setting -p 8 (LIME uses openmp for parallel processing). 

.. code-block:: bash

   lime -nSG -p 8 rt-lime.c 

The resulting line cubes (.fits) can be found on the data repository for this example.  

6. Let's create a new folder to host moment 0 maps and dendrograms.

.. code-block:: bash

   mkdir cube_products
   cd cube_products
   
7. Compute integrated intensity (moment 0) maps. Use the flag -i to specify the cloud inclination from ['faceon', 'edgeon', 'edgeon_phi90'] and -u for image units from ['jypxl', 'tau'] (defaults to 'faceon' and 'jypxl').

.. code-block:: bash

   python $PCAFACTORY/make_moment.py -i faceon
   python $PCAFACTORY/make_moment.py -i edgeon 
   python $PCAFACTORY/make_moment.py -i edgeon_phi90
   python $PCAFACTORY/make_moment.py -i edgeon_phi90 -u tau

Alternatively, the bash script *all.sh* included in the *src/* folder runs the script for all the inclinations and units using the -i and -u flags. 

.. code-block:: bash
   
   sh $PCAFACTORY/all.sh moment

The script executed by *all.sh* is determined by the accompanying argument in the command. You can use one from [moment, dendrogram, peaks, write, fit].  

8. Compute dendrograms on moment 0 maps to extract smaller-scale cloud portions.

.. code-block:: bash

   sh $PCAFACTORY/all.sh dendrogram

9. Get coordinates from moment 0 peaks in dendrogram leaves (30 pc wide boxes will be centred on these peaks later on for further analysis). Also, the following script creates the folder ./portions_moment0 where information from cloud portions, colour codes, PCA outputs and figures will stored. 

.. code-block:: bash

   sh $PCAFACTORY/all.sh peaks

10. Write cloud portions (30 pc wide boxes) in folder ./portions_moment0

.. code-block:: bash

   sh $PCAFACTORY/all.sh write
   cd portions_moment0

11. Run the principal component analysis (PCA) both for cloud portions and the cloud complex as a whole, and store the (PCA-derived) velocity fluctuations (dv) and spatial scales (l) in data files.

.. code-block:: bash

   sh $PCAFACTORY/run_pca.sh faceon
   sh $PCAFACTORY/run_pca.sh edgeon
   sh $PCAFACTORY/run_pca.sh edgeon_phi90
   sh $PCAFACTORY/run_pca.sh edgeon_phi90 tau

12. Read the PCA-derived scales to compute the cloud complex structure functions and show the resulting figures.

.. code-block:: bash

   sh $PCAFACTORY/all.sh fit

#python pca_summary.py  ??