## Prepare your model repo

Copy all three files (csa_bruteforce_params.ini, csa_bruteforce_params_def.py, csa_bruteforce_wf.py) to your repo. Change the parameters in csa_bruteforce_params.ini as needed. 

### These should be changed in csa_bruteforce_params.ini:

`model_name` set to your model name (this should have the same capitalization pattern as your model scripts, e.g. deepttc for deepttc_preprocess_improve.py, etc).

`epochs` set to max epochs appropriate for your model, or a low number for testing.

`uses_cuda_name` set to True if your model uses cuda_name as parameter, leave as False if it does not. Also set `cuda_name` if your model uses this.

### These you may want to change in csa_bruteforce_params.ini:

`csa_outdir` is `./bruteforce_output` but you can change to whatever directory you like.

`source_datasets`, `target_datasets`, and `split_nums` can be modified for testing purposes or quicker runs.


## Running workflow
1. Clone repo
```
git clone https://github.com/JDACS4C-IMPROVE/<YOUR_REPO>
```

2. Set up conda -- Follow instructions in your repo.

3. Activate conda -- Follow instructions in your repo.

4. Set up improve
```
source setup_improve.py
```
5. Run workflow
```
python csa_bruteforce_wf.py
```

Note: If submitting a job, steps 3-5 should be in the shell script.
