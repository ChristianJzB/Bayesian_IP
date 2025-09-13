
#import jax.numpy as jnp
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from Base.mcmc import MetropolisHastings,MCMCDA
from Base.lla import dgala

from elliptic_files.FEM_Solver import FEMSolver
from elliptic_files.elliptic import Elliptic
from elliptic_files.physics_informed_gp import EllipticPIGP,Elliptic1DPIGP


class EllipticMCMC(MetropolisHastings):
    """Subclass that defines the likelihoods for the Problem"""
    def __init__(self, surrogate, observation_locations, observations_values, nparameters=2, 
                 observation_noise=0.5, nsamples=1000000, burnin=None, proposal_type="pCN",
                 step_size=0.1,uniform_limit=1,my_reg = 1e-3, device="cpu", gp_marginal = True):
        
        super(EllipticMCMC, self).__init__(observation_locations =observation_locations, observations_values=observations_values, nparameters=nparameters, 
                 observation_noise=observation_noise, nsamples=nsamples, burnin=burnin, proposal_type=proposal_type, step_size=step_size,
                 uniform_limit=uniform_limit,my_reg=my_reg,device=device)
        
        self.surrogate = surrogate
        self.device = device
        self.gp_marginal = gp_marginal

        # Dictionary to map surrogate classes to their likelihood functions
        likelihood_methods = {
            FEMSolver: self.fem_log_likelihood, 
            Elliptic1DPIGP:self.gp_log_likelihood,
            Elliptic: self.nn_log_likelihood,
            dgala: self.dgala_log_likelihood}

        # Precompute the likelihood function at initialization
        surrogate_type = type(surrogate)
        if surrogate_type in likelihood_methods:
            self.log_likelihood_func = likelihood_methods[surrogate_type]
        else:
            raise ValueError(f"Surrogate of type {surrogate_type.__name__} is not supported.")


    def fem_log_likelihood(self, theta ):
        """
        Evaluates the log-likelihood given a FEM.
        """
        self.surrogate.theta = theta.cpu().numpy()  # Convert to numpy for FEM solver
        self.surrogate.solve()
        surg = self.surrogate.eval_at_points(self.observation_locations.cpu().numpy()).reshape(-1, 1)
        surg = torch.tensor(surg, device=self.device)
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))
    
    def gp_log_likelihood(self, theta):
        if self.gp_marginal:
            return self.gp_marginal_log_likelihood(theta)
        else:
            return self.gp_mean_log_likelihood(theta)
        
        
    def gp_marginal_log_likelihood(self,theta):
        """
        Evaluates the log-likelihood given a GP.
        """
        mean_surg, var_surg = self.surrogate.prediction(theta.reshape(1,-1),var=True)

        dy = mean_surg.shape[0]

        diff = (self.observations_values - mean_surg.reshape(-1, 1))
        sigma = var_surg +  torch.diag(torch.ones(dy) * self.observation_noise ** 2)

        L = torch.linalg.cholesky(sigma)
        k_inv_g = torch.linalg.solve(sigma,diff) 

        cte = 0.5 * (dy * torch.log(torch.tensor(2 * torch.pi)) + 2.0 * torch.sum(torch.log(torch.diag(L))))

        # Log-likelihood
        lg = -0.5 * torch.matmul(diff.T, k_inv_g)- cte
        return lg.squeeze()
    
    def gp_mean_log_likelihood(self,theta):
        """
        Evaluates the log-likelihood given a GP.
        """
        mean_surg = self.surrogate.prediction(theta.reshape(1,-1),var=False)
        diff = (self.observations_values - mean_surg.reshape(-1, 1))
        nll = -0.5 * torch.sum((diff ** 2) / (self.observation_noise ** 2))
        return nll


    def nn_log_likelihood(self, theta):
        """
        Evaluates the log-likelihood given a NN.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1).float()
        surg = self.surrogate.u(data.float()).detach()
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))

    def dgala_log_likelihood(self, theta):
        """
        Evaluates the log-likelihood given a dgala.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1).float()
        surg_mu, surg_sigma = self.surrogate(data)

        surg_mu = surg_mu.view(-1, 1)
        surg_sigma = surg_sigma[:, :, 0].view(-1, 1)

        sigma = self.observation_noise ** 2 + surg_sigma
        dy = surg_mu.shape[0]

        cte = 0.5 * (dy * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(sigma)))

        return -0.5 * torch.sum(((self.observations_values - surg_mu.reshape(-1, 1)) ** 2) / sigma)- cte
    
    def log_likelihood(self, theta):
        """Directly call the precomputed likelihood function."""
        return self.log_likelihood_func(theta)
        

class EllipticMCMCDA(MCMCDA):
    """Subclass that defines the likelihoods for the Problem"""

    def __init__(self,coarse_surrogate,finer_surrogate, observation_locations, observations_values,nparameters=2, 
                 observation_noise=0.5, iter_mcmc=1000000, iter_da = 20000,                 
                 proposal_type="pCN", uniform_limit=1, my_reg = 1e-3,step_size=0.1, device="cpu", gp_marginal = True):
        
        mcmc_instance = EllipticMCMC(surrogate = coarse_surrogate,observation_locations = observation_locations, 
                                    observations_values=observations_values,nparameters = nparameters, 
                                    observation_noise=observation_noise, nsamples = iter_mcmc,
                                    proposal_type = proposal_type,uniform_limit=uniform_limit, 
                                    my_reg=my_reg,step_size=step_size, device=device, gp_marginal = gp_marginal)
        
        super(EllipticMCMCDA, self).__init__(observation_locations, observations_values,mcmc_instance, nparameters, 
                 observation_noise, iter_mcmc, iter_da,proposal_type, uniform_limit, my_reg, step_size, device)
        
        self.coarse_surrogate = coarse_surrogate
        self.finer_surrogate = finer_surrogate
        self.device = device
        self.gp_marginal = gp_marginal

        # Dictionary to map surrogate classes to likelihood functions
        likelihood_methods = {
            FEMSolver: self.fem_log_likelihood,
            Elliptic1DPIGP:self.gp_log_likelihood,
            Elliptic: self.nn_log_likelihood,
            dgala: self.dgala_log_likelihood
        }

        # Precompute likelihood function for both surrogates
        self.log_likelihood_outer_func = self.get_likelihood_function(coarse_surrogate, likelihood_methods)
        self.log_likelihood_inner_func = self.get_likelihood_function(finer_surrogate, likelihood_methods)

    def fem_log_likelihood(self,surrogate, theta):
        """
        Evaluates the log-likelihood given a FEM.
        """
        surrogate.theta = theta.cpu().numpy()  # Convert to numpy for FEM solver
        surrogate.solve()
        surg = surrogate.eval_at_points(self.observation_locations.cpu().numpy()).reshape(-1, 1)
        surg = torch.tensor(surg, device=self.device)
        self.inner_likelihood_value = surg
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))

    def gp_log_likelihood(self,surrogate, theta):
        if self.gp_marginal:
            return self.gp_marginal_log_likelihood(surrogate,theta)
        else:
            return self.gp_mean_log_likelihood(surrogate,theta)
        
    def gp_mean_log_likelihood(self,surrogate,theta):
        """
        Evaluates the log-likelihood given a GP.
        """
        mean_surg = surrogate.prediction(theta.reshape(1,-1),var=False)

        self.outer_likelihood_value = mean_surg

        #dy = mean_surg.shape[0]

        diff = (self.observations_values - mean_surg.reshape(-1, 1))
        nll = -0.5 * torch.sum((diff ** 2) / (self.observation_noise ** 2))
        return nll
    
    def gp_marginal_log_likelihood(self,surrogate,theta):
        """
        Evaluates the log-likelihood given a GP.
        """
        mean_surg, var_surg = surrogate.prediction(theta.reshape(1,-1),var=True)

        self.outer_likelihood_value = mean_surg

        dy = mean_surg.shape[0]

        diff = (self.observations_values - mean_surg.reshape(-1, 1))

        sigma = var_surg +  torch.diag(torch.ones(dy) * self.observation_noise ** 2)

        L = torch.linalg.cholesky(sigma)
        k_inv_g = torch.linalg.solve(sigma,diff) 

        cte = 0.5 * (dy * torch.log(torch.tensor(2 * torch.pi)) + 2.0 * torch.sum(torch.log(torch.diag(L))))

        # Log-likelihood
        lg = -0.5 * torch.matmul(diff.T, k_inv_g)- cte
        return lg.squeeze()
    
    def nn_log_likelihood(self,surrogate,theta):
        """
        Evaluates the log-likelihood given a NN.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1).float()
        surg = surrogate.u(data.float()).detach()
        self.outer_likelihood_value = surg
        return -0.5 * torch.sum(((self.observations_values - surg) ** 2) / (self.observation_noise ** 2))

    def dgala_log_likelihood(self, surrogate,theta):
        """
        Evaluates the log-likelihood given a dgala.
        """
        data = torch.cat([self.observation_locations, theta.repeat(self.observation_locations.size(0), 1)], dim=1).float()
        surg_mu, surg_sigma = surrogate(data)

        surg_mu = surg_mu.view(-1, 1)
        surg_sigma = surg_sigma[:, :, 0].view(-1, 1)
        self.outer_likelihood_value = surg_sigma.detach()

        sigma = self.observation_noise ** 2 + surg_sigma
        dy = surg_mu.shape[0]

        cte = 0.5 * (dy * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(sigma)))
        return -0.5 * torch.sum(((self.observations_values - surg_mu.reshape(-1, 1)) ** 2) / sigma)- cte
    
    def get_likelihood_function(self, surrogate, likelihood_methods):
        """Precompute and return the appropriate likelihood function for a given surrogate."""
        for surrogate_type, likelihood_func in likelihood_methods.items():
            if isinstance(surrogate, surrogate_type):
                return lambda theta: likelihood_func(surrogate, theta)
        raise ValueError(f"Surrogate of type {type(surrogate).__name__} is not supported.")
    
    def log_likelihood_outer(self, theta):
        return self.log_likelihood_outer_func(theta)

    def log_likelihood_inner(self, theta):
        return self.log_likelihood_inner_func(theta)
    