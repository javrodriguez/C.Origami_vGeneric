srun --nodes=1 --ntasks-per-node=1 --time=24:00:00 --mem=50G --pty bash -i
module load singularity/3.11.5

#### Run maxATAC prepare inside Singularity container
singularity exec \
    -B ../maxATAC/:/maxATAC \ # this mounts maxATAC installation directory
    -B $(pwd):/mnt \
    --pwd /mnt \
    --no-home \
    maxatac.sif \
    maxatac prepare ...

#### Run maxATAC prepare inside Singularity container interactive shell
singularity shell \
    -B ../maxATAC/:/maxATAC \ # this mounts maxATAC installation directory
    -B $(pwd):/mnt \
    --pwd /mnt \
    --no-home \
    maxatac.sif
    
maxatac prepare ...
maxatac predict ...

exit