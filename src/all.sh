
filename=

case $1 in
    "moment")
	filename="make_moment.py"
	;;
    "dendro")
	filename="dendrogram_moment.py"
	;;
    "peaks")
	filename="get_peaks_leaves.py"
	;;
    "write")
	filename="write_portion.py"
	;;
    "fit")
	filename="new_fit_pca_points.py"
	;;
    *)
esac

python $TOOLS_PCA/$filename -i faceon
python $TOOLS_PCA/$filename -i edgeon
python $TOOLS_PCA/$filename -i edgeon_phi90
python $TOOLS_PCA/$filename -i edgeon_phi90 -u tau
