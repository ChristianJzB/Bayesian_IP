#!/bin/bash

# Define different values of N for experiments
N_VALUES=(1500)  # Adjust as needed
N_LAYERS=(3)  # Define hidden layers
N_BATCHES=(160 240 320 400 480)
KL_VALUES=(2 5 10 15 20) 

# Define default values for parameters
VERBOSE=""
TRAIN="--train"  # Set empty "" if you want default (False)
DEEPGALA="--deepgala"  # Empty means default (False)
NOISE_LEVEL="--noise_level 1e-3"
PROPOSAL="--pCN"
NN_MCMC=""  # Example: enabled
DGALA_MCMC=""
DA_MCMC_NN="--da_mcmc_nn"
DA_MCMC_DGALA="--da_mcmc_dgala"

for N in "${N_VALUES[@]}"; do
    for L in "${N_LAYERS[@]}"; do
        for BATCH in "${N_BATCHES[@]}"; do
            for KL in "${KL_VALUES[@]}"; do

            echo "Submitting job for N=$N with $L hidden layers"

        qsub -N "nv_N${N}_L${L}_BS${BS}" <<EOF
#!/bin/bash
#$ -cwd
#$ -q gpu
#$ -l gpu=1 
#$ -l h_vmem=40G
#$ -l h_rt=36:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python Navier-Stokes/experiments/nv_experiment.py --N $N --hidden_layers $L --num_neurons 300 --batch_size $BATCH --kl $KL $TRAIN $DEEPGALA $NOISE_LEVEL $PROPOSAL $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
EOF
            done
        done
    done
done
