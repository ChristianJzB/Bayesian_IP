import sys
import os
import torch
import wandb
import argparse
import numpy as np
from ml_collections import ConfigDict
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "Elliptic"))  # Explicitly add Elliptic folder

from Base.utilities import clear_hooks, Timer
from elliptic_files.utilities import generate_noisy_obs,run_mcmc_chain
from elliptic_files.FEM_Solver import FEMSolver

def elliptic_experiment():
    config = ConfigDict()
    config.verbose = False

    # Inverse problem parameters
    config.noise_level = 1e-4
    config.num_observations = 6
    config.fem_solver = 50
    config.KL_expansion = 2

    # MCMC configuration
    config.fem_mcmc = False
    config.repeat = 4

    config.proposal = "random_walk"
    config.proposal_variance = 1e-2
    config.uniform_limit = 1
    config.samples = 1_000_000
    config.FEM_h = 50

    return config

# Main experiment runner
def run_experiment(config_experiment,device):
    times = dict()

    # Generate noisy observations for Inverse Problem
    obs_points, sol_test = generate_noisy_obs(obs=config_experiment.num_observations,
                                              std=np.sqrt(config_experiment.noise_level),
                                              nparam=config_experiment.KL_expansion,
                                              vert=config_experiment.fem_solver)
    
    # MCMC FEM Samples (if enabled)
    print("Starting MCMC with FEM")
    
    for i in range(config_experiment.repeat):
        fem_solver = FEMSolver(np.zeros(config_experiment.KL_expansion), vert=config_experiment.FEM_h, M =config_experiment.KL_expansion )

        mcmcfem_timer = Timer(use_gpu=True)
        mcmcfem_timer.start()
        fem_samples = run_mcmc_chain(fem_solver, obs_points, sol_test, config_experiment, device, eval_val=True)
        tfemmcmc = mcmcfem_timer.stop()
        times["fem_mcmc"] = tfemmcmc
        np.save(f'./Elliptic/results/FEM_kl{config_experiment.KL_expansion}_var{config_experiment.noise_level}_{i}.npy', fem_samples[0])
        np.save(f'./Elliptic/results/FEM_eval_kl{config_experiment.KL_expansion}_var{config_experiment.noise_level}_{i}.npy', fem_samples[-1])


    # with open(f'./Elliptic/results/times/timing_FEM_{config_experiment.KL_expansion}.json', 'w') as f:
    #     json.dump(times, f, indent=4)  # indent for readability

# Main loop for different sample sizes
def main(verbose,kl,noise_level,proposal,repeat,device):
    
    config_experiment = elliptic_experiment()
    config_experiment.verbose = verbose
    config_experiment.KL_expansion = kl
    config_experiment.noise_level = noise_level
    config_experiment.proposal = proposal
    config_experiment.repeat = repeat

    run_experiment(config_experiment,device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Elliptic Experiment")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--kl", type=int,default=2,help="KL_expansion")
    parser.add_argument("--noise_level", type=float,default=1e-4,help="Noise level for IP")
    parser.add_argument("--proposal", type=str,default="pCN",help="MCMC Proposal")
    parser.add_argument("--repeat", type=int,default=4, help="Repeat MCMC ntimes")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Pass all arguments
    main(args.verbose,args.kl,args.noise_level, args.proposal,args.repeat,device)