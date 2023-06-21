#!/bin/bash
#SBATCH -J TST_shock
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --time=00:10:00
##SBATCH --reservation=openhack
#SBATCH --output main.output

module purge
module use --append /home/hakobyan/.modules
module load cudatoolkit/12.0
module load gcc-toolset/10 
module load hdf5/gcc-toolset-10/1.10.6 
module load adios2/gcc-10/hdf5-1.10.6 
module load kokkos/cuda-12/ampere-80
module load entity/gpu-ampere-80


#srun ./entity.xc -input /home/av8849/HACKATHON_entity/entity/src/pic/pgen/inputs/weibel.toml > entity.out
srun ./main.xc  > main.out


##!/bin/bash
##SBATCH -N     1
##SBATCH -t     12:00:00
##SBATCH -J     entrun
##SBATCH --mem  500G
##SBATCH --gres gpu:2

#EXECUTABLE=entity.xc
#INPUT=input.toml
#OUTPUT=output

###module load cudatoolkit/11.7
#module purge
#module use --append /home/hakobyan/.modules
#module load entity/gpu-ampere-80

##mkdir -p $OUTPUT
##cd $OUTPUT
##cp ../$INPUT .
##cp ../$EXECUTABLE .

###srun ./$EXECUTABLE -input $INPUT > report 2> error
#salloc -N 1 -t 12:00:00 -J entrun --mem=500G --gres=gpu:2
