#!/bin/sh

module load Miniconda/3_snakemake
module load Snakemake/7.8.5

dryrun=""
while getopts "p:j:r:R:u:n" opt
do
   case $opt in
       p) project_ID="$OPTARG" ;;
       j) max_nr_of_concurrent_jobs="$OPTARG" ;;
       r) resampling="$OPTARG" ;;
       R) rerun="-R $OPTARG" ;;
       u) until="--until $OPTARG" ;;
       n) dryrun="-n" ;;
   esac
done

if [ -z "$project_ID" ] || [ -z "$max_nr_of_concurrent_jobs" ] || [ -z "$resampling" ]
then
   echo "Usage: sh execute_on_cluster.sh -p project_ID -j max_nr_of_concurrent_jobs -r {bt | cv} [-R rule_to_rerun] [-u rule_to_stop_at] [-n]"
   exit 1
fi

current_time=$(date "+%Y.%m.%d")

mkdir -p logs.$current_time

type snakemake >/dev/null 2>&1 || { echo >&2 "I require snakemake but it is not installed or added to your path. Aborting..."; exit 1; }

snakemake $dryrun -s Snakefile_$resampling --config envfile='env_cluster.yaml' $rerun $until --rerun-incomplete --rerun-triggers mtime --jobs $max_nr_of_concurrent_jobs --use-conda --cluster-status "python cluster/statuscommand.py" --reason --jobscript cluster/jobscript.sh --cluster "qsub -A ${project_ID} -e logs.$current_time/{rule}.{wildcards}.{jobid}.errors -o logs.$current_time/{rule}.{wildcards}.{jobid}.output -l select=1:ncpus={params.cpus}:ngpus={params.gpus}:mem={params.mem}:arch=skylake -l walltime={params.walltime}"

# arch=skylake because there was a problem with different architectures (Ivy Bridge)
