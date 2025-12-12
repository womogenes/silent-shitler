#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mincpus 64
#SBATCH --mem 32000
#SBATCH --time 360

uv run generate_data.py
