#!/bin/bash

# Start multiple jobs.



#run_nums=( 5 )
num_runs=300

#for run in ${run_nums[@]}; do
for run in `seq 1 $num_runs`; do
    echo "Starting run $run"    
    qsub -N run$run -j oe -o logs/random_run_$run.txt -l nodes=1:ppn=1 -l walltime=1000:00:00 -q batch \
    -v run_path=`pwd`,cmd="/shared/users/asousa/software/anaconda2/bin/python random_run.py" run_job.pbs
done
#    -v run_path=`pwd`,cmd="/shared/users/asousa/software/anaconda2/bin/python run_$run.py" run_job.pbs
