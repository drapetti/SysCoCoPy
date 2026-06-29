#!/bin/sh
#PBS -W group_list=<enter_your_account_id>
#PBS -N job_parallel_template
#PBS -q <enter_your_preferred_queue>
#PBS -l <enter_your_job_specifications>

#Note: this script is to be submitted from the jobs folder where it resides

RUN=$PBS_JOBNAME
ID=$PBS_JOBID

cd $PBS_O_WORKDIR

#Parallel execution
seq 4 | parallel -j 4 -u --use-cpus-instead-of-cores "cd $PWD; ./$RUN.sh {} $RUN $ID"
