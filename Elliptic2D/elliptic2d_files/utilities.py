import numpy as np
from scipy.stats import qmc

import torch
from .train_elliptic2d import generate_data
from .FEM2d_Solver import Elliptic2D_FEM
from .elliptic2d_mcmc import EllipticMCMC, EllipticMCMCDA


def deepgala_data_fit(samples,nparameters,device,seed = 65647437836358831880808032086803839626):

    data_int, left_bc, right_bc,down_bc,up_bc = generate_data(samples, nparam=nparameters,seed=seed)
    data_int, left_bc, right_bc,down_bc,up_bc  = data_int.to(device),left_bc.to(device),right_bc.to(device),down_bc.to(device),up_bc.to(device) 
    
    dgala_data = {"data_fit": {"pde":data_int, "left_bc":left_bc,"right_bc":right_bc,"down_bc":down_bc,"up_bc":up_bc}, 
                "class_method": {"pde": ["elliptic_pde"], "left_bc":["u"],"right_bc":["u"],"down_bc":["u"],"up_bc":["u"]},
                "outputs": {"pde": ["elliptic"], "left_bc": ["ubcl"],"right_bc":["ubcr"],"down_bc":["ubcd"],"up_bc":["ubcu"]}}
    return dgala_data

def generate_noisy_obs(obs, nparam=2,ncells=50, mean=0, std=np.sqrt(1e-4), seed = 42):
    """
    Generates noisy observations for given parameters and observation points.
    """
    rng = np.random.default_rng(seed)

    obs_points = rng.uniform(low=0.0, high=1.0, size=(obs, 2))
    points_fem = np.hstack((obs_points, np.zeros((obs_points.shape[0], 1))))
    theta_thruth = rng.uniform(-1, 1, size=nparam)

    # Solve the FEM problem using the given theta values and roots
    solver = Elliptic2D_FEM(theta_thruth,ncells=ncells)
    solution = solver.solve()

    # Select the observation and solution points based on the indices
    sol_points = solver.evaluate_at_points(points_fem)

    # Generate noise and add it to the solution points
    noise_sol_points = add_noise(sol_points, mean, std)

    # Ensure proper reshaping of observation and solution points
    sol_test = noise_sol_points.reshape(-1, 1)

    return obs_points, sol_test,theta_thruth

def add_noise(solution, mean, std, seed = 0):
    """
    Adds Gaussian noise to the solution.
    """
    # rng = np.random.default_rng(seed)
    # noise = rng.normal(mean, std, solution.shape)
    np.random.seed(seed)
    noise = np.random.normal(mean, std, solution.shape)
    return solution + noise


def pigp_training_data_generation(theta_obs,spatial_points_obs,kl_expansion,noise_level,device,ncells=50,seed = 65647437836358831880808032086803839626):
    rng = np.random.default_rng(seed)

    obs_points, sol_test,_ = generate_noisy_obs(obs=spatial_points_obs,std=np.sqrt(noise_level),nparam=kl_expansion,ncells=ncells)

    thetas = rng.uniform(low = -1,high = 1,size=(theta_obs,kl_expansion))
    fem_solver = Elliptic2D_FEM(np.zeros(kl_expansion),ncells = ncells)
    # obs_points = rng.uniform(low=0.0, high=1.0, size=(spatial_points_obs, 2))
    points_fem = np.hstack((obs_points, np.zeros((obs_points.shape[0], 1))))

    training_data = np.zeros((theta_obs,spatial_points_obs ))
    for i,tht in enumerate(thetas):
        fem_solver.theta = tht
        
        fem_solver.solve()

        training_data[i,:] = fem_solver.evaluate_at_points(points_fem).reshape(1, -1)

    xf = np.array([
        [0.2, 0.2],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.8, 0.8],
        [0.5, 0.5],
        [0.2, 0.5],
        [0.5, 0.2],
        [0.8, 0.5],
        [0.5, 0.8],
        [0.35, 0.35]
    ])

    yf = 4 * xf[:, 0] * xf[:, 1]
    yf=yf.reshape(-1,1)

    grid = torch.linspace(0.1, 0.9, 5)

    # Bottom edge: y=0
    bottom = torch.stack([grid, torch.full_like(grid, 0)], dim=1)

    # Top edge: y=1
    top = torch.stack([grid, torch.full_like(grid, 1)], dim=1)

    # Left edge: x=0
    left = torch.stack([torch.full_like(grid, 0), grid], dim=1)

    # Right edge: x=1
    right = torch.stack([torch.full_like(grid, 1), grid], dim=1)

    # Concatenate all boundary points
    boundary_points = torch.cat([bottom, top, left, right], dim=0)

    y_bc = torch.zeros((boundary_points.shape[0],1))

    data_training = {"parameters_data": torch.tensor(thetas).to(device),
                    "solutions_data": torch.tensor(training_data).to(device),
                    "x_solutions_data":torch.tensor(obs_points).to(device),
                    "x_bc": torch.tensor(boundary_points).to(device),
                    "y_bc": torch.tensor(y_bc).to(device),
                    "source_func_x": torch.tensor(xf).to(device),
                    "source_func_f_x":torch.tensor(yf).to(device)
                    }
    return data_training,obs_points, sol_test

# Helper function to set up MCMC chain
def run_mcmc_chain(surrogate_model, obs_points, sol_test, config_experiment,device, gp_marginal=False):
    mcmc = EllipticMCMC(
        surrogate=surrogate_model,
        observation_locations=obs_points,
        observations_values=sol_test,
        observation_noise=np.sqrt(config_experiment.noise_level),
        nparameters=config_experiment.KL_expansion,
        nsamples=config_experiment.samples,
        proposal_type=config_experiment.proposal,
        step_size=config_experiment.proposal_variance,
        uniform_limit = config_experiment.uniform_limit,
        device=device,
        gp_marginal = gp_marginal
    )
    return mcmc.run_chain(verbose=config_experiment.verbose)

def run_da_mcmc_chain(nn_surrogate_model,fem_solver, obs_points, sol_test, config_experiment,device, gp_marginal=False):
    elliptic_mcmcda =  EllipticMCMCDA(nn_surrogate_model,fem_solver, 
                        observation_locations= obs_points, observations_values = sol_test, 
                        nparameters=config_experiment.KL_expansion,
                        observation_noise=np.sqrt(config_experiment.noise_level),
                        iter_mcmc=config_experiment.iter_mcmc, iter_da = config_experiment.iter_da,
                        proposal_type=config_experiment.proposal,
                        uniform_limit = config_experiment.uniform_limit,
                        step_size=config_experiment.proposal_variance, 
                        device=device, gp_marginal = gp_marginal )   
    return elliptic_mcmcda.run_chain(verbose=config_experiment.verbose)


