#!/usr/bin/bash


#PBS -N acmm_30
#PBS -j oe -l ngpus=1
#PBS -q GPU-1A
#PBS -o pbs_infer-sp.log
#PBS -e pbs_error-sp.log
#PBS -M s2320014@jaist.ac.jp
#PBS -m e

source ~/.bashrc
source activate base
conda activate acmm
SOURCE_PATH=/home/s2320014/acmmm25_mmfc

cd $SOURCE_PATH

python fine_tune.py