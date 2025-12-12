#!/bin/bash

#SBATCH -p mit_normal_gpu
#SBATCH --mincpus 32
#SBATCH --mem 64000
#SBATCH --time 360

uv run generate_data.py
