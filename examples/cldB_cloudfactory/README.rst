
Cloud Complex description
-------------------------

* In this example we use a self-gravitating cloud complex influenced by a large-scale Milky Way-like galactic potential.
* Local gravitational forces are turned on for 2 Myr in a high resolution portion of the simulation, from which the cloud complex is extracted. 
* There are also supernova explosions randomly distributed all over the Galaxy.
* There is also chemical evolution of CO and Hydrogen species; sink particles representing star-forming regions; radiative heating and cooling and galactic differential rotation.

Find full details of the simulation setup of this and other types of cloud complexes in the Cloud Factory series of papers (I, II)

These simulations are powered by a customised version of the AREPO code (Springel+2010)

pcafactory-data repository
--------------------------

The data required for this example can be downloaded `here <https://girder.hub.yt/#user/5da06b5868085e00016c2dee/folder/5da06ef668085e00016c2df3>`_,
where you will find the following files:
 
* Simulation snapshot of the cloud complex.
* 12CO J=1-0 intensity cubes: 3 line intensity cubes for different cloud orientations (face-on, edge-on phi=0, edge-on phi=90) generated with sf3dmodels and LIME.
* Optical depth cube for the edge-on phi=90 case.

Quick Tutorial
--------------

1. Download the simulation snapshot 
   
.. code-block:: bash

   curl https://girder.hub.yt/api/v1/item/5da0777768085e00016c2e01/download -o Restest_extracted_001_240

2. Read and clean the snapshot, and save the formatted physical information of the cloud for radiative transfer with LIME.

.. code-block:: bash
      
   python make_arepo_lime.py

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/cellsize_numdens-AREPOgrid.png?raw=true
   :width: 30%

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/3Dpoints_snap.png?raw=true
   :width: 30%

3. The output files are stored in the folder ./Subgrids by default

.. code-block:: bash
   
   cd Subgrids

4. Download the CO excitation information from the LAMDA database. 

.. code-block:: bash
   
   curl https://home.strw.leidenuniv.nl/~moldata/datafiles/co.dat -o co.dat 

5. Run our version of LIME, adapted to model radiative transfer of Arepo-like (voronoi) meshes. It is available `here <https://github.com/andizq/star-forming-regions>`_. The flag -S indicates that the grid was generated with `sf3dmodels <https://github.com/andizq/star-forming-regions>`_, and the flag -G indicates that the input grid is not uniform. The flag -n is to show log messages in terminal. We use 8 cores by setting -p 8 (LIME uses openmp for parallel processing). 

.. code-block:: bash

   lime -nSG -p 8 rt-lime.c 

The resulting line cubes (.fits) can be found in the repository prepared for this example.

6. Let's create a new folder to store moment 0 maps and dendrograms.

.. code-block:: bash

   mkdir cube_products
   cd cube_products
   
7. Compute integrated intensity (moment 0) maps. Use the flag -i to specify the cloud inclination from ['faceon', 'edgeon', 'edgeon_phi90'] and -u for image units from ['jypxl', 'tau'] (defaults to 'faceon' and 'jypxl').

.. code-block:: bash

   python $PCAFACTORY/make_moment.py -i faceon
   python $PCAFACTORY/make_moment.py -i edgeon 
   python $PCAFACTORY/make_moment.py -i edgeon_phi90
   python $PCAFACTORY/make_moment.py -i edgeon_phi90 -u tau

Alternatively, the bash script *run_all.sh* included in the *src/* folder runs the script for all the inclinations and units.

.. code-block:: bash
   
   sh $PCAFACTORY/run_all.sh moment

The script executed by *run_all.sh* is determined by the accompanying argument in the command. You can pick one from [moment, dendrogram, peaks, write, fit].  

8. Compute dendrograms on moment 0 maps to extract smaller-scale cloud portions.

.. code-block:: bash

   sh $PCAFACTORY/run_all.sh dendrogram

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/img_moment0dendro_jypxl_faceon.png?raw=true

.. note::

   The file *pars_dendrogram.txt* allows handling dendrogram parameters for all cloud orientations and/or cube units without modifying the source scripts. 
   The script *make_dendrogram.py* (executed by **run_all.sh dendrogram**) uses these parameters to run the function Dendrogram.compute() from `astrodendro <https://dendrograms.readthedocs.io>`_   
:: 

   # inclination   delta_factor    min_npix 
   faceon		1	     180
   edgeon		5	     150
   edgeon_phi90		10	     150
   edgeon_phi90tau	1	     70


9. The following script finds the coordinates of zeroth moment peaks in dendrogram leaves and centres 30 pc boxes on them for the principal component analysis. It creates the folder *./portions_moment0* to store information from these cloud portions and PCA outputs.

.. code-block:: bash

   sh $PCAFACTORY/run_all.sh peaks

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/img_moment0_jypxl_faceon.png?raw=true


10. Extract cloud portion cubes from the cloud complex cube (.fits) into *./portions_moment0* using the 30 pc boxes.

.. code-block:: bash

   sh $PCAFACTORY/run_all.sh write
   cd portions_moment0

11. Run the principal component analysis (PCA) both for the cloud portions and for the cloud complex as a whole, and store the (PCA-derived) velocity fluctuations (dv) and spatial scales (l) in data files.

.. code-block:: bash

   sh $PCAFACTORY/run_pca.sh faceon
   sh $PCAFACTORY/run_pca.sh edgeon
   sh $PCAFACTORY/run_pca.sh edgeon_phi90
   sh $PCAFACTORY/run_pca.sh edgeon_phi90 tau

.. note::
   The file *pars_pca.txt* controls the parameter *min_eigval* for cloud portions and the cloud complex as a whole for all orientations and/or cube units. The parameter *min_eigval* sets the minimum percentage of variance considered for the PCA study. High percentages are ideal to keep as much information as possible but too high values may lead to clustering of PCA-derived scales around spatial/spectral resolution limits. See further details of this parameter on `turbustat.statistics.PCA <https://turbustat.readthedocs.io/en/latest/api/turbustat.statistics.PCA.html#turbustat.statistics.PCA.compute_pca>`_
  
::

   # incl	min_eigval_portion	min_eigval_cloud
   faceon		0.999		0.999
   edgeon		0.999		0.999
   edgeon_phi90		0.999		0.999
   edgeon_phi90tau 	0.999		0.999


12. Read the PCA-derived scales to compute the cloud complex structure functions and show the resulting figures.

.. code-block:: bash

   sh $PCAFACTORY/run_all.sh fit

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/img_fit_jypxl_faceon_allportions.png?raw=true

.. image:: https://github.com/andizq/andizq.github.io/blob/master/pcafactory-data/examples-data/cldB_cloudfactory/PCA_jypxl_faceon_offsets.png?raw=true

.. note::
   With the file *overlaped_portions.txt* you can control which cloud portion(s) should be removed from the analysis using the cloud id. Commonly, one might want to reject cloud portions that overlap one another too much and also those where the PCA-derived scales are too few (which can lead to unreliable fits).

::

   #Overlaping portions to reject. if None put -1
   faceon	 	0,4,7,14,15
   edgeon	 	9,5,2,3,7
   edgeon_phi90	 	0,3,5
   edgeon_phi90tau	 2,4

(Missing: docs for plot_line_portions.py, plot_column.py, pca_summary.py)