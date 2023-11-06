# Multi-Output-Workflow

Welcome! This is the git repository of the paper "A Critical Review of Multi-Output Support Vector Regression".

To run the workflow on a local machine, please do the following:

First, install [Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) if you haven't already.  
Then, you can set the configurations like the files to process, solvers to use, hyperparameters, etc. in config.yaml.  
After done configurating, you can either run  
`snakemake -s Snakefile_bt --use-conda --cores <number of cores or 'all' (without quotation marks)>` for bootstrapping or  
`snakemake -s Snakefile_cv --use-conda --cores <number of cores or 'all' (without quotation marks)>` for nested cross-validation.  
(If the command `snakemake` is not recognized, you may need to activate your snakemake environment with `conda activate snakemake`.)

To run the workflow on a cluster, please do the following:

Change `cluster/jobscript.sh` and `env_cluster.yaml` according to your cluster.  
Then, you can set the configurations like the files to process, solvers to use, hyperparameters, etc. in config.yaml.  
After done configurating, you can run  
`sh execute_on_cluster.sh -p project_ID -j max_nr_of_concurrent_jobs -r {bt | cv} [-R rule_to_rerun] [-u rule_to_stop_at] [-n (for a dry run)]`.
