(Under development)

Cloud Complex description
-------------------------

* For this example we use a self-gravitating cloud complex inmersed in the galactic potential of a Milky Way-type galaxy. 
* We let the object evolve for 2 Myr under local gravitational forces in a high resolution portion of the simulation mesh. 
* There are random supernova explosions located all across the density distribution of the galaxy.
* There is also chemical evolution for CO and Hydrogen species; sink particle formation (stellar systems); radiative heating and cooling and galactic differential rotation.

For full details on the simulation setup of this (cloud complex B) and other types of cloud complexes see the papers I and II of the Cloud Factory's series: Smith et al. subm. and Izquierdo et al. in prep (https://github.com/andizq/andizq.github.io/tree/master/pcafactory-data). 

pcafactory-data repository
--------------------------

The data required for this example are on https://girder.hub.yt/#user/5da06b5868085e00016c2dee/folder/5da06ef668085e00016c2df3.

There you can find the following files:
 
* Simulation snapshot of the cloud complex.
* 12CO J=1-0 intensity cubes: 3 line intensity cubes for different cloud orientations (face-on, edge-on phi=0, edge-on phi=90) generated with sf3dmodels and LIME (https://github.com/andizq/star-forming-regions).
* Optical depth cube for the edge-on phi=90 case.

Quick Tutorial
--------------

#. Download the simulation snapshot # Restest
#. Read the snapshot and save physical information for running LIME.

.. code-block:: bash
      
   python make_arepo_lime.py [3D grid distribution .png]

#. cd Subgrids/
#. curl https://home.strw.leidenuniv.nl/~moldata/datafiles/co.dat -o co.dat  # Download the CO file from the LAMDA database. 
#. lime -nSG -p 8 rt-lime.c # The resulting line cubes can be found on the data repository for this example (here).  

   bla blah

#. cd Dendrograms_portions/
#. python make_moment.py [.pngs]
#. python dendrogram.py [.pngs]
#. python get_peaks_leaves.py [.pngs]
#. python write_portion.py
#. cd portions_moment0/
#. python exmp_PCA.py
#. python new_fits_pca.py [.pngs]
#. python pca_summary.py  ??