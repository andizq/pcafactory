
filename=

case $1 in
    "moment")
	filename="make_moment.py"
	;;
    "dendrogram")
	filename="make_dendrogram.py"
	;;
    "peaks")
	filename="get_peaks_leaves.py"
	;;
    "write")
	filename="write_portion.py"
	;;
    "fit")
	filename="make_fit_pca.py"
	;;
    *)
esac

python $PCAFACTORY/$filename -i faceon
python $PCAFACTORY/$filename -i edgeon
python $PCAFACTORY/$filename -i edgeon_phi90
python $PCAFACTORY/$filename -i edgeon_phi90 -u tau
