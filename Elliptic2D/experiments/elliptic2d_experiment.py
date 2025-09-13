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
sys.path.append(os.path.join(project_root, "Elliptic2D"))  # Explicitly add Elliptic folder

from Base.lla import dgala
from Base.utilities import clear_hooks, Timer
from elliptic2d_files.train_elliptic2d import train_elliptic2d,generate_data,samples_param
from elliptic2d_files.utilities import *
from elliptic2d_files.FEM2d_Solver import Elliptic2D_FEM


def elliptic2d_experiment():
    config = ConfigDict()
    config.verbose = False

    # NN training config
    config.train = False

    # Weights & Biases
    config.wandb = ConfigDict()
    config.wandb.project = "Elliptic2D-IP"
    config.wandb.name = "MDNN"
    config.wandb.tag = None

    # Model settings
    config.nn_model = "MDNN"
    config.lambdas = {"elliptic": 1, "ubcl": 1, "ubcr": 1,"ubcd":1,"ubcu":1}
    config.model = ConfigDict()
    config.model.input_dim = 4
    config.model.hidden_dim = 100
    config.model.num_layers = 3
    config.model.out_dim = 1
    config.model.activation = "tanh"
    config.KL_expansion = 2

    # Periodic embeddings
    #config.model.period_emb = ConfigDict({"period":(1.0, 1.0), "axis":(0, 1) })

    # Fourier embeddings
    config.model.fourier_emb = ConfigDict({"embed_scale":1,"embed_dim":100,"exclude_last_n":2})

    # Training settings
    config.seed = 42
    config.learning_rate = 0.001
    config.decay_rate = 0.95
    config.epochs = 5_000
    config.start_scheduler = 0.5
    config.scheduler_step = 50
    config.nn_samples = 200
    config.batch_size = 250
    config.weights_update = 250
    config.alpha = 0.9  # For updating loss weights

    # DeepGala
    config.deepgala = False

    # Inverse problem parameters
    config.noise_level = 1e-4
    config.num_observations = 6
    config.fem_solver = 75

    # MCMC configuration
    config.fem_mcmc = False
    config.nn_mcmc = False
    config.dgala_mcmc = False
    config.repeat = 4

    config.proposal = "random_walk"
    config.proposal_variance = 1e-3
    config.uniform_limit = 1
    config.samples = 1_000_000
    config.FEM_h = 50

    # Delayed Acceptance
    config.da_mcmc_nn = False
    config.da_mcmc_dgala = False
    config.iter_mcmc = 1_000_000
    config.iter_da = 1_00_000

    return config


# Main experiment runner
def run_experiment(config_experiment,device):

    # Step 1: Training Phase (if needed)
    model_specific = f"_hl{config_experiment.model.num_layers}_nn{config_experiment.model.hidden_dim}_s{config_experiment.nn_samples}_bs{config_experiment.batch_size}_kl{config_experiment.KL_expansion}"
    path_nn_model = f"./Elliptic2D/models/elliptic2d"+model_specific+".pth"
    path_dgala_model = f"./Elliptic2D/models/elliptic2d_dgala"+model_specific+".pth"
    fem_path = f'./Elliptic2D/results/e2dFEM_kl{config_experiment.KL_expansion}_var{config_experiment.noise_level}.npy'
    times = dict()

    if config_experiment.train:
        print(f"Running training with {config_experiment.nn_samples} samples...")
        config_experiment.wandb.name = "elliptic2d" + model_specific
        config_experiment.model.input_dim = 2 + config_experiment.KL_expansion
        config_experiment.model.fourier_emb["embed_dim"] = config_experiment.model.hidden_dim
        config_experiment.model.fourier_emb["exclude_last_n"] = config_experiment.KL_expansion

        train_timer = Timer(use_gpu=True)
        train_timer.start()
        pinn_nvs = train_elliptic2d(config_experiment, device=device)
        ttrain = train_timer.stop()
        times["train"] = ttrain
        print(f"Completed training with {config_experiment.nn_model} samples.")

    # Step 2: Fit deepGALA
    if config_experiment.deepgala:
        print(f"Starting DeepGaLA fitting for NN_s{config_experiment.nn_samples}")
        nn_surrogate_model = torch.load(path_nn_model)
        nn_surrogate_model.eval()

        dgala_timer = Timer(use_gpu=True)
        dgala_timer.start()
        data_fit = deepgala_data_fit(config_experiment.nn_samples,config_experiment.KL_expansion,device)
        llp = dgala(nn_surrogate_model)
        llp.fit(data_fit)
        llp.optimize_marginal_likelihoodb()
        clear_hooks(llp)
        tdagala = dgala_timer.stop()
        times["deepgala"] = tdagala

        torch.save(llp, path_dgala_model)

    # Step 3: Generate noisy observations for Inverse Problem
    obs_points, sol_test,_ = generate_noisy_obs(obs=config_experiment.num_observations,
                                              std=np.sqrt(config_experiment.noise_level),
                                              nparam=config_experiment.KL_expansion,
                                              ncells=config_experiment.fem_solver)
    
    # Step 4: Neural Network Surrogate for MCMC
    if config_experiment.nn_mcmc:
        print(f"Starting MCMC with NN_s{config_experiment.nn_samples}")
        nn_surrogate_model = torch.load(path_nn_model, map_location=device)
        nn_surrogate_model.eval()

        mcmc_timer = Timer(use_gpu=True)
        mcmc_timer.start()
        nn_samples = run_mcmc_chain(nn_surrogate_model, obs_points, sol_test, config_experiment,device)
        tmcmc = mcmc_timer.stop()
        times["nn_mcmc"] = tmcmc

        np.save('./Elliptic2D/results/NN'+ model_specific+ f'_var{config_experiment.noise_level}.npy', nn_samples[0])
    
    # Step 5: DeepGaLA Surrogate for MCMC
    if config_experiment.dgala_mcmc:
        print(f"Starting MCMC with DeepGaLA_s{config_experiment.nn_samples}")
        llp = torch.load(path_dgala_model, map_location=device)
        llp.model.set_last_layer("output_layer")  # Re-register hooks
        llp._device = device

        mcmcdg_timer = Timer(use_gpu=True)
        mcmcdg_timer.start()
        nn_samples = run_mcmc_chain(llp, obs_points, sol_test, config_experiment, device)
        tdgmcmc = mcmcdg_timer.stop()
        times["dgala_mcmc"] = tdgmcmc

        np.save('./Elliptic2D/results/dgala'+model_specific+ f'_var{config_experiment.noise_level}.npy', nn_samples[0])
    
    # Step 6: MCMC FEM Samples (if enabled)
    if config_experiment.fem_mcmc:
        print("Starting MCMC with FEM")
        fem_solver = Elliptic2D_FEM(np.zeros(config_experiment.KL_expansion), ncells=config_experiment.FEM_h)

        mcmcfem_timer = Timer(use_gpu=True)
        mcmcfem_timer.start()
        fem_samples = run_mcmc_chain(fem_solver, obs_points, sol_test, config_experiment, device)
        tfemmcmc = mcmcfem_timer.stop()
        times["fem_mcmc"] = tfemmcmc

        np.save(fem_path, fem_samples[0])

    # Step 7: Delayed Acceptance for NN
    if config_experiment.da_mcmc_nn:
        print(f"Starting MCMC-DA with NN_s{config_experiment.nn_samples} and FEM")
        
        nn_surrogate_model = torch.load(path_nn_model, map_location=device)
        nn_surrogate_model.eval()

        fem_solver = Elliptic2D_FEM(np.zeros(config_experiment.KL_expansion), ncells=config_experiment.FEM_h)

        for i in range(config_experiment.repeat):
            results = run_da_mcmc_chain(nn_surrogate_model,fem_solver,obs_points, sol_test, config_experiment, device)
            
            nn_samples, acceptance_res,proposal_thetas,lh_val_nn,lh_val_solver,tmcmc,tda = results
            times["nn_mcmc"] = tmcmc
            times["nn_da"] = tda

            np.save('./Elliptic2D/results/NN'+ model_specific+ f'_var{config_experiment.noise_level}_{i}.npy', nn_samples)
            np.save('./Elliptic2D/results/mcmc_da_nn' +model_specific+ f'_{config_experiment.noise_level}_{i}.npy', acceptance_res)
            np.save("./Elliptic2D/results/mcmc_da_nn_proposal_thetas" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", proposal_thetas)
            np.save("./Elliptic2D/results/mcmc_da_nn_lh_nn" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", lh_val_nn)
            np.save("./Elliptic2D/results/mcmc_da_nn_lh_solver" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", lh_val_solver)

    # Step 8: Delayed Acceptance for Dgala
    if config_experiment.da_mcmc_dgala:
        print(f"Starting MCMC-DA with DGALA_s{config_experiment.nn_samples} and FEM")
        llp = torch.load(path_dgala_model, map_location=device)
        llp.model.set_last_layer("output_layer")  # Re-register hooks
        llp._device = device

        fem_solver = Elliptic2D_FEM(np.zeros(config_experiment.KL_expansion), ncells=config_experiment.FEM_h)
        for i in range(config_experiment.repeat):
            results  = run_da_mcmc_chain(llp,fem_solver,obs_points, sol_test, config_experiment, device)
            nn_samples, acceptance_res,proposal_thetas,lh_val_nn,lh_val_solver,tmcmc,tda = results

            times["dgala_mcmc"] = tmcmc
            times["dgala_da"] = tda
            np.save('./Elliptic2D/results/dgala'+model_specific+ f'_var{config_experiment.noise_level}_{i}.npy', nn_samples)
            np.save('./Elliptic2D/results/mcmc_da_dgala' +model_specific+f'_{config_experiment.noise_level}_{i}.npy', acceptance_res)
            np.save("./Elliptic2D/results/mcmc_da_dgala_proposal_thetas" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", proposal_thetas)
            np.save("./Elliptic2D/results/mcmc_da_dgala_lh_nn" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", lh_val_nn)
            np.save("./Elliptic2D/results/mcmc_da_dgala_lh_solver" + model_specific + f"_var{config_experiment.noise_level}_{i}.npy", lh_val_solver)

    with open(f'./Elliptic2D/results/times/timing_nn'+ model_specific+ '.json', 'w') as f:
        json.dump(times, f, indent=4)  # indent for readability

# Main loop for different sample sizes
def main(verbose,N,hidden_layers,num_neurons,batch_size,kl,train,deepgala, 
         noise_level,proposal,fem_mcmc,nn_mcmc,dgala_mcmc,da_mcmc_nn,da_mcmc_dgala, repeat,device):
    
    config_experiment = elliptic2d_experiment()
    config_experiment.verbose = verbose
    config_experiment.nn_samples = N 
    config_experiment.model.num_layers = hidden_layers
    config_experiment.model.hidden_dim = num_neurons
    config_experiment.batch_size = batch_size
    config_experiment.KL_expansion = kl
    config_experiment.train = train
    config_experiment.deepgala = deepgala
    config_experiment.noise_level = noise_level
    config_experiment.proposal = proposal
    config_experiment.fem_mcmc = fem_mcmc
    config_experiment.nn_mcmc = nn_mcmc
    config_experiment.dgala_mcmc = dgala_mcmc
    config_experiment.da_mcmc_nn = da_mcmc_nn
    config_experiment.da_mcmc_dgala = da_mcmc_dgala
    config_experiment.repeat = repeat

    run_experiment(config_experiment,device)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Elliptic Experiment")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--N", type=int, default=100, help="Number of training samples")
    parser.add_argument("--hidden_layers", type=int,default=2,help="Number of layers")
    parser.add_argument("--num_neurons", type=int,default=20,help="Number of neurons/layer")
    parser.add_argument("--batch_size", type=int,default=10,help="Mini batche size")
    parser.add_argument("--kl", type=int,default=2,help="KL_expansion")
    parser.add_argument("--train", action="store_true", help="Train NN")
    parser.add_argument("--deepgala", action="store_true", help="Fit DeepGala")
    parser.add_argument("--noise_level", type=float,default=1e-4,help="Noise level for IP")
    parser.add_argument("--proposal", type=str,default="random_walk",help="MCMC Proposal")
    parser.add_argument("--fem_mcmc", action="store_true", help="Run MCMC for FEM")
    parser.add_argument("--nn_mcmc", action="store_true", help="Run MCMC for NN")
    parser.add_argument("--dgala_mcmc", action="store_true", help="Run MCMC for dgala")
    parser.add_argument("--da_mcmc_nn", action="store_true", help="Run DA-MCMC for NN")
    parser.add_argument("--da_mcmc_dgala", action="store_true", help="Run DA-MCMC for DeepGala")
    parser.add_argument("--repeat", type=int,default=4, help="Repeat MCMC ntimes")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(os.getcwd())

    # Pass all arguments
    main(args.verbose, args.N,args.hidden_layers,args.num_neurons,args.batch_size,args.kl, args.train, 
         args.deepgala, 
         args.noise_level, args.proposal, 
         args.fem_mcmc, args.nn_mcmc, args.dgala_mcmc, 
         args.da_mcmc_nn, args.da_mcmc_dgala, args.repeat, device)