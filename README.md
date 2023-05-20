# VHH (V->jj, H->bb) Analysis

## `singularity`
run analysis

    singularity run \
    -B /cvmfs:/cvmfs:ro \
    -B /uscms/home/chuyuanl/nobackup/VHH:/VHH/input:ro \
    -B <output dir>:/VHH/output \
    -B <repo dir>:/analysis \
    docker://chuyuanliu/heptools:latest \
    python /analysis/run.py

## `conda`
set up and activate conda environment

    conda env create -f local.yml
    conda activate vhh

# TODO
- JEC
- JCM reweighting
- run SvB NN, FvT NN, kl BDT
- skim and save from NanoAOD
- submit condor job
- test on data and tt MC