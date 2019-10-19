ARG1=${1:-faceon}
ARG2=${2:-jypxl}
ARG3=${3:-nonLTE}

f=$(ls *img_"$ARG2"_"$ARG1"_portion* | wc -l)
echo $f

for((i=0;i<f;i++)); do
    python $PCAFACTORY/make_pca.py -n $i -i $ARG1 -u $ARG2 -R $ARG3
done
python $PCAFACTORY/make_pca.py -c 1 -i $ARG1 -u $ARG2 -R $ARG3
