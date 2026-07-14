!#/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --ntasks-per-cpu=1
#SBATCH --time=01:00:00

module purge
module load julia/1.11.7

julia eg-intro.jl
