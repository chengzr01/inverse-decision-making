#!/bin/bash
#SBATCH --job-name=sandbox          # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G            # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=01:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # send email when job begins

module purge
module load proxy/default
module load anaconda3/2025.12
conda activate gym
python myscript.py