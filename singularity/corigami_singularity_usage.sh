srun --nodes=1 --ntasks=1 --cpus-per-task=1 --time=12:00:00 --mem=50Gb --gres=gpu:a100:1 --pty bash
module load singularity/3.11.5

#### Run C.Origami inside Singularity container interactive shell
singularity shell \
    -B ../C.Origami:/C.Origami \                      # this mounts the C.Origami directory (necessary only with corigami_dependencies.sif)
    -B ../C.Origami/corigami_data:/corigami_data \    # this mounts the corigami_data directory
    -B ../input_bw/:/input_bw \                       # this mounts the input bw directory 
    -B $(pwd):/mnt \
    --pwd /mnt \
    --no-home \
    --cleanenv \
    corigami_dependencies.sif

#### If using the corigami.sif image, which includes the C.Origami installation
corigami-predict ...

exit

#### If using the corigami_dependencies.sif image, C.Origami needs to be installed installed
# install your mounted C.Origami version inside a virtual environment
cd /C.Origami
python3 -m venv venv
source /C.Origami/venv/bin/activate
pip install .

# activate the virtual environment and run corigami-predict
source /C.Origami/venv/bin/activate
corigami-predict ...
deactivate

exit