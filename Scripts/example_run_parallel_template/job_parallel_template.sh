#!/bin/sh

RUN=$2
ID=$3
DATE=`date +%y-%m-%d_%H%M`
RUN_DIR=<enter_your_SysCoCoPy_path>
JOBS_DIR=<enter_your_jobs_path>/${RUN}/run_$1

#Set to load modules
source <enter_modules_shell_file>

#Set up the environment
miniconda() {
        module use -a /swbuild/analytix/tools/modulefiles
        module load miniconda3/v4
    }
miniconda

#Set up each parallel run
mkdir -p ${JOBS_DIR}
cd ${JOBS_DIR}

echo "Executing run $1 on" `hostname` "in $PWD"

cd ${RUN_DIR}

#Set the log file dir and name
LOG_FILE=${JOBS_DIR}/${RUN}-id-${ID}.log

#Python with colored traceback
copython() {
        python $@ 2>&1 | sed -e "s/Traceback/${boldyellowonblue}&${norm}/g" \
        -e "s/File \".*\.py\".*$/${boldyellow}&${norm}/g" \
        -e "s/\, line [[:digit:]]\+/${boldred}&${norm}/g"
    }

## Set path for user's poetry environment
export PATH="<enter_your_path>"

#Run SysCoCoPy
echo Begin time: `date`
/usr/bin/time -p poetry run python SysCoCoPy/Scripts/example_run_parallel_template/run-Correctors_Comparer-parameters-${RUN}.py $1 | tee $LOG_FILE
echo End time: `date`
