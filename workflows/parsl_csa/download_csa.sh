data_dir="csa_data"
if [ ! -d $PWD/$data_dir/ ]; then
    echo "Download CSA data"
    wget --cut-dirs=8 -P ./ -nH -np -m --reject "index.html*,index.*" https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/
else
    echo "CSA data folder already exists"
fi