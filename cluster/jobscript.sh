#!/bin/bash

# load modules for cluster execution if a module system exists on the HPC used for computation
# else specify in env_cluster.yaml

module load Miniconda/3_snakemake
module load Snakemake/7.8.5

{exec_job}
