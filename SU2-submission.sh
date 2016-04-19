#!/bin/bash
### specify the queue and that job is not re-runable,
### as well as the resources and the name
#PBS -q cuth
#PBS -r n
#PBS -l nodes=1,walltime=200:00:00
#PBS -N SU2
#PBS -o /home/jt2798/cuth-log
#PBS -e /home/jt2798/cuth-log

date

/home/jt2798/SU2-Pure-Gauge/SU2Gauge

date

