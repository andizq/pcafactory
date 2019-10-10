(Under development)

Example cloud
-------------

* For this example we use a self-gravitating cloud complex inmersed in the galactic potential of a Milky Way-type galaxy. 
* We let the object evolve for 2 Myr under local gravitational forces in a high resolution portion of the simulation mesh. 
* There are random supernova explosions located all across the density distribution of the galaxy.
* There is also chemical evolution for CO and Hydrogen species; sink particle formation (stellar systems); radiative heating and cooling and galactic differential rotation.

For full details on the simulation setup of this and other types of cloud complexes see the papers I and II of the Cloud Factory's series: Smith et al. subm. and Izquierdo et al. in prep (https://github.com/andizq/andizq.github.io/tree/master/pcafactory-data). 

pcafactory data repository
--------------------------

The data repository of this example is on https://github.com/andizq/andizq.github.io/tree/master/pcafactory-data/examples-data/cldB_cloudfactory

There you can find the following files:
 
* Simulation snapshot of the cloud complex.
* 12CO J=1-0 intensity: 3 line cubes for different cloud orientations (face-on, edge-on phi=0, edge-on phi=90) generated with sf3dmodels and LIME (https://github.com/andizq/star-forming-regions).
* Optical depth cube for the edge-on phi=90 case.

Quick Tutorial
--------------

1. Download the simulation snapshot
2. python make_arepo_lime.py [3D grid distribution .png]
3. cd Subgrids/
4. lime -nSG -p 8 rt-lime.c # The resulting line cubes can be found on the data repository for this example (here).  
5. cd Dendrograms_portions/
6. python make_moment.py [.pngs]
7. python dendrogram.py [.pngs]
8. python get_peaks_leaves.py [.pngs]
9. python write_portion.py
10. cd portions_moment0/
11. python exmp_PCA.py
12. python new_fits_pca.py [.pngs]
13. python pca_summary.py  ??