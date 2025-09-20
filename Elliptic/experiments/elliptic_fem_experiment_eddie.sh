#!/bin/bash

# Define different values of N for experiments
CONFIG_MODELS=(2 3 4 5)

# Define default values for parameters (change as needed)
VERBOSE=""
NOISE_LEVEL="--noise_level 1e-4"
PROPOSAL="--proposal pCN" # Default RMMCMC

for KL in "${CONFIG_MODELS[@]}"; do
    echo "Submitting FEM for KL exp $KL"

    qsub -N "FEM_elliptic_kl$KL" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -l h_vmem=40G
#$ -l h_rt=6:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python Elliptic/experiments/elliptic_FEM.py --kl $KL $NOISE_LEVEL $PROPOSAL
EOF
done