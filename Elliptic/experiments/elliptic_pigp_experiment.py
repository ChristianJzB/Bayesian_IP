import sys
import os
import torch
import argparse
import numpy as np
from ml_collections import ConfigDict
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "Elliptic"))  # Explicitly add Elliptic folder

from Base.utilities import Timer
from elliptic_files.utilities import generate_noisy_obs, pigp_training_data_generation,run_da_mcmc_chain,run_mcmc_chain
from elliptic_files.FEM_Solver import FEMSolver
from elliptic_files.physics_informed_gp import Elliptic1DPIGP

def elliptic_pigp_experiment():
    config = ConfigDict()
    config.verbose = False
    config.train_gp = False
    # PIGP training config
    config.KL_expansion = 2
    config.observed_solutions = 10
    config.observed_spatial_points = 6

    # Inverse problem parameters
    config.noise_level = 1e-4
    config.num_observations = 6
    config.fem_solver = 50

    config.proposal = "pCN"
    config.proposal_variance = 1e-2
    config.uniform_limit = 1
    config.samples = 1_000_000
    config.FEM_h = 50

    # Delayed Acceptance
    config.da_mcmc_nn = False
    config.da_mcmc_dgala = False
    config.iter_mcmc = 1_000_000
    config.iter_da = 50_000
    config.repeat = 4

    return config


# Main experiment runner
def run_experiment(config_experiment,device):

    # Step 1: Training Phase (if needed)
    model_specific = f"_spatial{config_experiment.observed_spatial_points}_nsol{config_experiment.observed_solutions}_kl{config_experiment.KL_expansion}"
    times = dict()
    
    if config_experiment.train_gp:
        print(f"Training GP_s{config_experiment.observed_solutions}")

        data_gen = Timer(use_gpu=True)
        data_gen.start()
        data_training = pigp_training_data_generation(config_experiment.observed_solutions,config_experiment.observed_spatial_points,config_experiment.KL_expansion,device)
        dgene = data_gen.stop()
        times["data_gen"] = dgene

        train_timer = Timer(use_gpu=True)
        train_timer.start()
        elliptic_gp = Elliptic1DPIGP(data_training,device=device)
        elliptic_gp.train_gp()
        elliptic_gp.optimize_mll()
        ttrain = train_timer.stop()
        times["train"] = ttrain

        torch.save(elliptic_gp, f"./Elliptic/models/elliptic_pigp_{model_specific}.pt")
    else:
        elliptic_gp = torch.load(f"./Elliptic/models/elliptic_pigp_{model_specific}.pt", map_location=device)


    # Step 3: Generate noisy observations for Inverse Problem
    obs_points, sol_test = generate_noisy_obs(obs=config_experiment.num_observations,
                                              std=np.sqrt(config_experiment.noise_level),
                                              nparam=config_experiment.KL_expansion,
                                              vert=config_experiment.fem_solver)
    
    # Step 4: Mean MCMCM
    if config_experiment.gp_mcmc:
        print(f"Starting MCMC with GP_s{config_experiment.observed_solutions}")

        mcmc_timer = Timer(use_gpu=True)
        mcmc_timer.start()
        nn_samples = run_mcmc_chain(elliptic_gp, obs_points, sol_test, config_experiment,device,False)
        np.save('./Elliptic/results/PIGP_mean'+ model_specific+ f'_var{config_experiment.noise_level}.npy', nn_samples[0])
        tmcmc = mcmc_timer.stop()
        times["gp_mean_mcmc"] = tmcmc

        # Step 4: Mean MCMCM
    if config_experiment.gp_mcmc:
        print(f"Starting MCMC with GP_s{config_experiment.observed_solutions}")

        mcmc_timer = Timer(use_gpu=True)
        mcmc_timer.start()
        nn_samples = run_mcmc_chain(elliptic_gp, obs_points, sol_test, config_experiment,True,device)
        np.save('./Elliptic/results/PIGP_marginal'+ model_specific+ f'_var{config_experiment.noise_level}.npy', nn_samples[0])
        tmcmc = mcmc_timer.stop()
        times["gp_marginal_mcmc"] = tmcmc

    # Step 7: Delayed Acceptance for GP_mean
    if config_experiment.da_mcmc_gp_mean:
        print(f"Starting MCMC-DA with GP_s{config_experiment.observed_solutions} and FEM for {config_experiment.repeat}times")

        fem_solver = FEMSolver(np.zeros(config_experiment.KL_expansion), vert=config_experiment.FEM_h,M = config_experiment.KL_expansion)

        for i in range(config_experiment.repeat):
            results = run_da_mcmc_chain(elliptic_gp,fem_solver,obs_points, sol_test, config_experiment,device,False)
            gp_samples, acceptance_res,proposal_thetas,lh_val_nn,lh_val_solver,tmcmc,tda = results

            times["gp_mean_mcmc"] = tmcmc
            times["gp_mean_da"] = tda
            
            np.save('./Elliptic/results/PIGP_mean'+ model_specific+ f'_var{config_experiment.noise_level}_{i}.npy', gp_samples)
            np.save('./Elliptic/results/mcmc_da_pigp_mean' +model_specific+ f'_{config_experiment.noise_level}_{i}.npy', acceptance_res)
            np.save("./Elliptic/results/mcmc_da_pigp_mean_proposal_thetas" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", proposal_thetas)
            np.save("./Elliptic/results/mcmc_da_pigp_mean_lh_nn" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", lh_val_nn)
            np.save("./Elliptic/results/mcmc_da_pigp_mean_lh_solver" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", lh_val_solver)

        # Step 7: Delayed Acceptance for GP_mean
    if config_experiment.da_mcmc_gp_marginal:
        print(f"Starting MCMC-DA with GP_s{config_experiment.observed_solutions} and FEM for {config_experiment.repeat}times")

        fem_solver = FEMSolver(np.zeros(config_experiment.KL_expansion), vert=config_experiment.FEM_h,M = config_experiment.KL_expansion)

        for i in range(config_experiment.repeat):
            results = run_da_mcmc_chain(elliptic_gp,fem_solver,obs_points, sol_test, config_experiment,device,True)
            gp_samples, acceptance_res,proposal_thetas,lh_val_nn,lh_val_solver,tmcmc,tda = results

            times["gp_marginal_mcmc"] = tmcmc
            times["gp_margianl_da"] = tda
            
            np.save('./Elliptic/results/PIGP_marginal'+ model_specific+ f'_var{config_experiment.noise_level}_{i}.npy', gp_samples)
            np.save('./Elliptic/results/mcmc_da_pigp_marginal' +model_specific+ f'_{config_experiment.noise_level}_{i}.npy', acceptance_res)
            np.save("./Elliptic/results/mcmc_da_pigp_marginal_proposal_thetas" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", proposal_thetas)
            np.save("./Elliptic/results/mcmc_da_pigp_marginal_lh_nn" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", lh_val_nn)
            np.save("./Elliptic/results/mcmc_da_pigp_marginal_lh_solver" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", lh_val_solver)


    with open(f'./Elliptic/results/times/timing_gp'+ model_specific+ '.json', 'w') as f:
        json.dump(times, f, indent=4)  # indent for readability

# Main loop for different sample sizes
def main(verbose,N,spatial_points,kl,train_gp,noise_level,proposal,gp_mcmc,da_mcmc_gp_mean,da_mcmc_gp_marginal,repeat,device):
    
    config_experiment = elliptic_pigp_experiment()
    config_experiment.verbose = verbose
    config_experiment.observed_solutions = N 
    config_experiment.observed_spatial_points = spatial_points
    config_experiment.KL_expansion = kl
    config_experiment.train_gp = train_gp
    config_experiment.noise_level = noise_level
    config_experiment.proposal = proposal
    config_experiment.gp_mcmc = gp_mcmc
    config_experiment.da_mcmc_gp_mean = da_mcmc_gp_mean
    config_experiment.da_mcmc_gp_marginal = da_mcmc_gp_marginal
    config_experiment.repeat = repeat

    run_experiment(config_experiment,device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Elliptic Experiment")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--N", type=int, required=True, help="Number of oberved solutions")
    parser.add_argument("--spatial_points", type=int,default=6,help="Number of spatial points for training")
    parser.add_argument("--kl", type=int,default=2,help="KL_expansion")
    parser.add_argument("--train_gp", action="store_true", help="Train GP")
    parser.add_argument("--noise_level", type=float,default=1e-4,help="Noise level for IP")
    parser.add_argument("--proposal", type=str,default="random_walk",help="MCMC Proposal")
    parser.add_argument("--gp_mcmc", action="store_true", help="Run MCMC for NN")
    parser.add_argument("--da_mcmc_gp_mean", action="store_true", help="Run DA-MCMC for NN")
    parser.add_argument("--da_mcmc_gp_marginal", action="store_true", help="Run DA-MCMC for NN")
    parser.add_argument("--repeat", type=int,default=4, help="Repeat MCMC ntimes")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(os.getcwd())

    # Pass all arguments
    main(args.verbose, args.N,args.spatial_points,args.kl, args.train_gp,
         args.noise_level, args.proposal,
         args.gp_mcmc, args.da_mcmc_gp_mean, args.da_mcmc_gp_marginal,args.repeat,device)