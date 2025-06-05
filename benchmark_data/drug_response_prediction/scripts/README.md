# IMPROVE DRP Benchmark Data
Provides the standard benchmark data download and the option to use coderdata

## For the standard benchmark data
Run the following to download the standard benchmark data from ftp.

```bash
source download_csa.sh
```

## For coderdata DRP data
Create a working directory and run the following commands to format the coderdata data into IMPROVE directory and file structure

```bash
python prepare_data_for_improve.py setup -w <WORKING_DIR>
python prepare_data_for_improve.py download -w <WORKING_DIR>
python prepare_data_for_improve.py process -w <WORKING_DIR>
```

Note that we recommend using the (-e 'SMI_55373,SMI_55606,SMI_17810,SMI_23544,SMI_55398,SMI_55337,SMI_55347,SMI_55464,SMI_22647,SMI_56600,SMI_22812') arugment with the process command, particulary with models that use RDKit (e.g. GraphDRP), to remove problematic drugs so that the process command is as follows: 

```bash
python prepare_data_for_improve.py process -w <WORKING_DIR> -e 'SMI_55373,SMI_55606,SMI_17810,SMI_23544,SMI_55398,SMI_55337,SMI_55347,SMI_55464,SMI_22647,SMI_56600,SMI_22812'
```
