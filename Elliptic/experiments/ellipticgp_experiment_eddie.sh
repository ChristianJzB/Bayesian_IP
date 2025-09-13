# #!/bin/sh

# # Define different values of N for experiments
# N_VALUES=(10 25 50 75) 
# KL_VALUES=(2 3 4 5 6) 

# # Define default values for parameters (change as needed)
# TRAIN_GP="--train_gp"
# NOISE_LEVEL="--noise_level 1e-4"
# PROPOSAL="--proposal pCN" # Default RMMCMC
# GP_MCMC=""
# DA_MCMC_GP="--da_mcmc_gp"

# for N in "${N_VALUES[@]}"; do
#             for KL in "${KL_VALUES[@]}"; do
#                  echo "Submitting job for GP_N=$N and KL exp $KL"

#     qsub -N "elliptic_GP_N${N}" <<EOF
# #!/bin/bash
# #$ -cwd
# # -q gpu
# # -l gpu=1 
# #$ -l h_vmem=40G
# #$ -l h_rt=36:00:00 

# # Load necessary modules
# . /etc/profile.d/modules.sh
# module load cuda/12.1
# module load miniforge
# conda activate experiments

# # Run the experiment with dynamic and fixed arguments
# python Elliptic/experiments/elliptic_pigp_experiment.py --N $N --kl $KL $TRAIN_GP $NOISE_LEVEL $PROPOSAL $GP_MCMC $DA_MCMC_GP
# EOF

#     done
# done

#!/bin/sh

# Define different values of N for experiments

CONFIG_MODELS=("110 3" "110 4" "110 5")

# Define default values for parameters (change as needed)
TRAIN_GP="--train_gp"
NOISE_LEVEL="--noise_level 1e-4"
PROPOSAL="--proposal pCN" # Default RMMCMC
GP_MCMC=""
DA_MCMC_GP_MEAN="--da_mcmc_gp"
DA_MCMC_GP_MARGINAL=""


for CONFIG in "${CONFIG_MODELS[@]}"; do
    read N KL <<< "$CONFIG"

    echo "Submitting job for N=$N and KL exp $KL"

    qsub -N "elliptic_GP_N${N}" <<EOF
#!/bin/bash
#$ -cwd
# -q gpu
# -l gpu=1 
#$ -l h_vmem=40G
#$ -l h_rt=36:00:00 

# Load necessary modules
. /etc/profile.d/modules.sh
module load cuda/12.1
module load miniforge
conda activate experiments

# Run the experiment with dynamic and fixed arguments
python Elliptic/experiments/elliptic_pigp_experiment.py --N $N --kl $KL $TRAIN_GP $NOISE_LEVEL $PROPOSAL $GP_MCMC $DA_MCMC_GP_MEAN $DA_MCMC_GP_MARGINAL
EOF
done