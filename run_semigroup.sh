#!/bin/bash

################ Slurm header (adapted from course script) ################
#SBATCH --account=large-sc-2
#SBATCH --time=00:20:00
#SBATCH --job-name=sg-attn
#SBATCH --output=/iopsstor/scratch/cscs//Megatron-LM-new-attention/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-new-attention/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --environment=/iopsstor/scratch/cscs/ooikonomou/ngc_pt_jan.toml

#!/bin/bash

# Minimal Slurm script to run semigroup_fast.py and write logs
# next to this file. Any Pyxis errors you see after this are
# coming from the cluster/container setup, not from this script.

#SBATCH --account=large-sc-2
#SBATCH --time=00:20:00
#SBATCH --job-name=sg-attn
#SBATCH --output=/users/ooikonomou/Megatron-LM-new-attention/sg-attn-%j.out
#SBATCH --error=/users/ooikonomou/Megatron-LM-new-attention/sg-attn-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
# Use the same environment file as your earlier working jobs
# (this must be fixed on the cluster side if the image path is bad).
#SBATCH --environment=/iopsstor/scratch/cscs/ooikonomou/ngc_pt_jan.toml
#SBATCH --no-requeue

echo "START TIME: $(date)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run from the directory where sbatch was called (recommended: scratch clone)
REPO_DIR="$SLURM_SUBMIT_DIR"

# Make sure we run python from that repo directory *inside* the container
CMD="cd $REPO_DIR && python3 semigroup_fast.py"
echo "Running: $CMD" >&2

srun --cpus-per-task=$SLURM_CPUS_PER_TASK bash --norc --noprofile -c "$CMD"

echo "END TIME: $(date)"
echo "[${HOSTNAME}] CMD: $CMD" | tee "$DEBUG_DIR/cmd.txt"
