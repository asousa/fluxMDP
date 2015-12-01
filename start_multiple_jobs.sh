#!/bin/bash

# Start multiple jobs.



run_nums=( 6 )


for run in ${run_nums[@]}; do
    echo "Starting run $run"    
    qsub -N run$run -j oe -o logs/run_$run.txt -l nodes=1:ppn=2 -l walltime=1000:00:00 -q batchnew \
    -v run_path=`pwd`,cmd="/shared/users/asousa/software/anaconda2/bin/python run_$run.py" run_job.pbs
done
#    -v run_path=`pwd`,cmd="/shared/users/asousa/software/anaconda2/bin/python run_$run.py" run_job.pbs
