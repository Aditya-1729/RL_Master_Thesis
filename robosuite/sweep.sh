#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=wandb_sweep
#SBATCH --output=/hpcwork/ru745256/master_thesis/7_8/robosuite/robosuite/task_out.%J.out
#SBATCH --error=/hpcwork/ru745256/master_thesis/7_8/robosuite/robosuite/error_task_out.%J.out
#SBATCH --nodes=1 # request one nodes
#SBATCH --cpus-per-task=8  # ask for 2 cpus per task
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --account=rwth1272

export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"


# load modules
# module switch intel gcc      
conda activate robosuite

# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark/rlkit':${PYTHONPATH}
# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark':${PYTHONPATH}
PYTHONPATH='/hpcwork/ru745256/master_thesis/7_8/robosuite':${PYTHONPATH}
# PYTHONPATH='/hpcwork/ru745256/master_thesis/30_6/robosuite/stable-baselines3':${PYTHONPATH}                       
export PYTHONPATH
export MUJOCO_GL='disabled'
#export PYOPENGL_PLATFORM=osmesa
#export DISPLAY=guilinuxbox:0.0


wandb agent master_thesis_ap/HPC_sweep/9lznwav2 --count=700