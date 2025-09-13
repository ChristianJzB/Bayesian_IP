# #!/bin/sh

# # Define different values of N for experiments
# N_VALUES=(4000) 
# NUM_LAYERS=(2)
# NUM_NEURONS=(100)
# KL_VALUES=(5 10 15 20) 
# N_BATCHES=(100 160 200 250 400 500)

# # Define default values for parameters (change as needed)
# VERBOSE=""
# TRAIN="--train"  # Set empty "" if you want default (False)
# DEEPGALA=""      # Empty means default (False)
# NOISE_LEVEL="--noise_level 1e-4"
# PROPOSAL="--proposal pCN" # Default RMMCMC
# FEM_MCMC="--fem_mcmc"
# NN_MCMC=""  # Example: enabled
# DGALA_MCMC=""
# DA_MCMC_NN="--da_mcmc_nn"
# DA_MCMC_DGALA=""

# # Flag to check if FEM_MCMC has been added
# FEM_MCMC_ADDED=true

# for N in "${N_VALUES[@]}"; do
#     for NLAYER in "${NUM_LAYERS[@]}"; do
#         for NNEURON in "${NUM_NEURONS[@]}"; do
#             for BATCH in "${N_BATCHES[@]}"; do
#                 for KL in "${KL_VALUES[@]}"; do
#                     echo "Submitting job for N=$N and KL exp $KL"

#                     # Add FEM_MCMC only for the first iteration
#                     if [ "$FEM_MCMC_ADDED" = false ]; then
#                         FEM_MCMC_FLAG="--fem_mcmc"
#                         FEM_MCMC_ADDED=true
#                     else
#                         FEM_MCMC_FLAG=""
#                     fi

#     qsub -N "elliptic_N${N}" <<EOF
# #!/bin/bash
# #$ -cwd
# # -q gpu
# # -l gpu=1 
# #$ -l h_vmem=40G
# #$ -l h_rt=6:00:00 

# # Load necessary modules
# . /etc/profile.d/modules.sh
# module load cuda/12.1
# module load miniforge
# conda activate experiments

# # Run the experiment with dynamic and fixed arguments
# python Elliptic/experiments/elliptic_experiment.py --N $N --hidden_layers $NLAYER --num_neurons $NNEURON --batch_size $BATCH --kl $KL $TRAIN $DEEPGALA $NOISE_LEVEL $PROPOSAL $FEM_MCMC_FLAG $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
# EOF             
#                 done
#             done
#         done
#     done
# done

#!/bin/sh

# Define different values of N for experiments

CONFIG_MODELS=("2 100 4000 250 3" "2 100 4000 250 4" "2 100 4000 250 5")

# Define default values for parameters (change as needed)
VERBOSE=""
TRAIN=""  # Set empty "" if you want default (False)
DEEPGALA=""      # Empty means default (False)
NOISE_LEVEL="--noise_level 1e-4"
PROPOSAL="--proposal pCN" # Default RMMCMC
FEM_MCMC="--fem_mcmc"
NN_MCMC=""  # Example: enabled
DGALA_MCMC=""
DA_MCMC_NN="--da_mcmc_nn"
DA_MCMC_DGALA=""

# Flag to check if FEM_MCMC has been added
FEM_MCMC_ADDED=true

for CONFIG in "${CONFIG_MODELS[@]}"; do
    read NLAYER NNEURON N BATCH KL <<< "$CONFIG"

    echo "Submitting job for N=$N and KL exp $KL"

    # Add FEM_MCMC only for the first iteration
    if [ "$FEM_MCMC_ADDED" = false ]; then
        FEM_MCMC_FLAG="--fem_mcmc"
        FEM_MCMC_ADDED=true
    else
        FEM_MCMC_FLAG=""
    fi

    qsub -N "elliptic_N${N}" <<EOF
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
python Elliptic/experiments/elliptic_experiment.py --N $N --hidden_layers $NLAYER --num_neurons $NNEURON --batch_size $BATCH --kl $KL $TRAIN $DEEPGALA $NOISE_LEVEL $PROPOSAL $FEM_MCMC_FLAG $NN_MCMC $DGALA_MCMC $DA_MCMC_NN $DA_MCMC_DGALA
EOF
done