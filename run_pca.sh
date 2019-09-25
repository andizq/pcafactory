f=$(ls *img_jypxl_faceon_portion* | wc -l)
echo $f
for((i=0;i<f;i++)); do
    python3 $TOOLS_PCA/exmp_PCA.py -n $i
done
python3 $TOOLS_PCA/exmp_PCA.py -c 1

e=$(ls *img_jypxl_edgeon_portion* | wc -l)
echo $e
for((i=0;i<e;i++)); do
    python3 $TOOLS_PCA/exmp_PCA.py -i edgeon -n $i
done
python3 $TOOLS_PCA/exmp_PCA.py -i edgeon -c 1

ep=$(ls *img_jypxl_edgeon_phi90_portion* | wc -l)
echo $ep
for((i=0;i<ep;i++)); do
    python3 $TOOLS_PCA/exmp_PCA.py -i edgeon_phi90 -n $i
done
python3 $TOOLS_PCA/exmp_PCA.py -i edgeon_phi90 -c 1

ept=$(ls *img_tau_edgeon_phi90_portion* | wc -l)
echo $ept
for((i=0;i<ept;i++)); do
    python3 $TOOLS_PCA/exmp_PCA.py -i edgeon_phi90 -n $i -u tau
done
python3 $TOOLS_PCA/exmp_PCA.py -i edgeon_phi90 -c 1 -u tau
