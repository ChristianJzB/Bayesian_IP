import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np

from typing import Tuple, Callable, Optional
from .deep_models import Dense
import inspect
import time
from scipy.optimize import fsolve, brentq
from scipy.stats import entropy


class BaseSampler(Dataset):
    def __init__(self, batch_size, rng_seed=1234):
        """
        Base class for samplers.
        
        :param batch_size: The size of the batch to be sampled.
        :param rng_seed: Random seed for reproducibility.
        """
        self.batch_size = batch_size
        self.rng_seed = rng_seed
        self.num_devices = torch.cuda.device_count()  # Gets the number of GPUs

    def __getitem__(self, index):
        """
        Generate one batch of data.

        :param index: Index for batch sampling, unused here but needed for Dataset API.
        :return: Batch of data
        """
        # Increment the seed (or set to something dynamic like current time)
        if self.rng_seed:        
            self.rng_seed += index
        batch = self.data_generation()
        return batch

    def data_generation(self):
        """
        Abstract method to generate data, to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")


class UniformSampler(BaseSampler):
    def __init__(self, dom, batch_size, device,rng_seed=None):
        super().__init__(batch_size, rng_seed)
        self.device = device
        self.dom = dom.to(self.device)  # (dim, 2), where each row is [min, max] for a dimension
        self.dim = dom.shape[0]
        self.use_seed = rng_seed is not None

    def data_generation(self):
        """
        Generates batch_size random samples uniformly within the domain, respecting the RNG seed.
        
        :return: Tensor of shape (batch_size, dim)
        """
        # Generate random samples uniformly within the given domain (min, max)
        min_vals = self.dom[:, 0]
        max_vals = self.dom[:, 1]
        
        # Initialize random number generator with seed for reproducibility
        if self.use_seed:
            torch.manual_seed(self.rng_seed)  # Reset the seed for each batch
        
        # Generate random samples in the range [min_vals, max_vals] for each dimension
        # Using rand to create uniform samples in the [0, 1) range
        rand_vals = torch.rand(self.batch_size, self.dim).to(self.device)
        
        # Scale the values to be in the [min, max] range for each dimension
        batch = min_vals + rand_vals * (max_vals - min_vals)

        return batch
    

def stat_ar(x, every=2000):
    split = x.shape[0] // every
    means_every = np.mean(x[:split * every].reshape(split, every), axis=1)  # Reshape and calculate means
    mean = np.mean(means_every)
    std = np.std(means_every)
    return mean, std, means_every


def get_decorated_methods(cls,decorator):
    methods = inspect.getmembers(cls, predicate=inspect.ismethod)
    decorated_methods = [name for name, method in methods if getattr(method, decorator, False)]
    return decorated_methods

def clear_hooks(model):
    for module in model.modules():
        module._forward_hooks.clear()
        module._backward_hooks.clear()

def histogram_(x, bins=100):
    # Calculate histogram
    counts, bin_edges = np.histogram(x, bins=bins)

    # Normalize counts to form a probability density
    counts = counts / (sum(counts) * np.diff(bin_edges))

    # Calculate the bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, counts


class FeatureExtractor(nn.Module):
    """Feature extractor for a PyTorch neural network.
    A wrapper which can return the output of the penultimate layer in addition to
    the output of the last layer for each forward pass. If the name of the last
    layer is not known, it can determine it automatically. It assumes that the
    last layer is linear and that for every forward pass the last layer is the same.
    If the name of the last layer is known, it can be passed as a parameter at
    initilization; this is the safest way to use this class.
    Based on https://gist.github.com/fkodom/27ed045c9051a39102e8bcf4ce31df76.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model
    last_layer_name : str, default=None
        if the name of the last layer is already known, otherwise it will
        be determined automatically.
    """
    def __init__(self, model: nn.Module, last_layer_name: Optional[str] = None) -> None:
        super().__init__()
        self.model = model
        self._features = dict()
        if last_layer_name is None:
            self.last_layer = None
        else:
            self.set_last_layer(last_layer_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. If the last layer is not known yet, it will be
        determined when this function is called for the first time.

        Parameters
        ----------
        x : torch.Tensor
            one batch of data to use as input for the forward pass
        """
        if self.last_layer is None:
            # if this is the first forward pass and last layer is unknown
            out = self.find_last_layer(x)
        else:
            # if last and penultimate layers are already known
            out = self.model(x)
        return out

    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass which returns the output of the penultimate layer along
        with the output of the last layer. If the last layer is not known yet,
        it will be determined when this function is called for the first time.

        Parameters
        ----------
        x : torch.Tensor
            one batch of data to use as input for the forward pass
        """
        out = self.forward(x)
        features = self._features[self._last_layer_name]
        return out, features

    def set_last_layer(self, last_layer_name: str) -> None:
        """Set the last layer of the model by its name. This sets the forward
        hook to get the output of the penultimate layer.

        Parameters
        ----------
        last_layer_name : str
            the name of the last layer (fixed in `model.named_modules()`).
        """
        # set last_layer attributes and check if it is linear
        self._last_layer_name = last_layer_name
        self.last_layer = dict(self.model.named_modules())[last_layer_name]
        if not isinstance(self.last_layer, (nn.Linear, Dense)):
            raise ValueError('Use model with a linear last layer.')

        # set forward hook to extract features in future forward passes
        self.last_layer.register_forward_hook(self._get_hook(last_layer_name))

    def _get_hook(self, name: str) -> Callable:
        def hook(_, input, __):
            # only accepts one input (expects linear layer)
            self._features[name] = input[0].detach()
        return hook

    def find_last_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Automatically determines the last layer of the model with one
        forward pass. It assumes that the last layer is the same for every
        forward pass and that it is an instance of `torch.nn.Linear`.
        Might not work with every architecture, but is tested with all PyTorch
        torchvision classification models (besides SqueezeNet, which has no
        linear last layer).

        Parameters
        ----------
        x : torch.Tensor
            one batch of data to use as input for the forward pass
        """
        if self.last_layer is not None:
            raise ValueError('Last layer is already known.')

        act_out = dict()
        def get_act_hook(name):
            def act_hook(_, input, __):
                # only accepts one input (expects linear layer)
                try:
                    act_out[name] = input[0].detach()
                except (IndexError, AttributeError):
                    act_out[name] = None
                # remove hook
                handles[name].remove()
            return act_hook

        # set hooks for all modules
        handles = dict()
        for name, module in self.model.named_modules():
            handles[name] = module.register_forward_hook(get_act_hook(name))

        # check if model has more than one module
        # (there might be pathological exceptions)
        if len(handles) <= 2:
            raise ValueError('The model only has one module.')

        # forward pass to find execution order
        out = self.model(x)

        # find the last layer, store features, return output of forward pass
        keys = list(act_out.keys())
        for key in reversed(keys):
            layer = dict(self.model.named_modules())[key]
            if len(list(layer.children())) == 0:
                self.set_last_layer(key)

                # save features from first forward pass
                self._features[key] = act_out[key]

                return out

        raise ValueError('Something went wrong (all modules have children).')


class Timer:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if self.use_gpu:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = None
            self.end_time = None

    def start(self):
        if self.use_gpu:
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def stop(self):
        if self.use_gpu:
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) / 1000.0  # Convert ms to seconds
        else:
            self.end_time = time.perf_counter()
            return self.end_time - self.start_time


def compute_seq_pairs(N_KL, include_00=False):
    trunc_Nx = int(np.floor(np.sqrt(2 * N_KL)) + 1)
    pairs = []
    # Generate (i, j) pairs and their squared norms
    for i in range(trunc_Nx):
        for j in range(trunc_Nx):
            if i == 0 and j == 0 and not include_00:
                continue
            pairs.append([i, j, i**2 + j**2])

    # Sort pairs by the squared norm (third column) and select the first N_KL pairs
    pairs = sorted(pairs, key=lambda x: x[2])[:N_KL]
    
    # Return only the (i, j) pairs, discarding the norm
    return np.array(pairs)[:, :2]


class RootFinder:
    def __init__(self, lam , M, equation=None):
        """
        lam: parameter lambda (for your equation)
        M: number of intervals to search for roots
        equation: optional, if you want to provide a custom equation
        """
        self.lam = lam
        self.M = M
        self.c = 1 / lam
        self.equation = equation if equation else self.default_equation

    def default_equation(self, x):
        """Default transcendental equation: tan(x) = (2c*x)/(x^2 - c^2)"""
        c = self.c
        return np.tan(x) - (2 * c * x) / (x**2 - c**2)

    def find_roots(self):
            """Find the roots using the Brent or fsolve method depending on the case."""
            roots = []
            for i in range(self.M):
                wmin = (i - 0.499) * np.pi
                wmax = (i + 0.499) * np.pi

                # Handle the singularity around c
                if wmin <= self.c <= wmax:  
                    if wmin > 0:
                        root = fsolve(self.equation, (self.c + wmin) / 2)[0]
                        roots.append(root)
                    root = fsolve(self.equation, (self.c + wmax) / 2)[0]
                    roots.append(root)
                elif wmin > 0:  
                    root = brentq(self.equation, wmin, wmax)
                    roots.append(root)
            
            return np.array(roots)
    

def discrete_kl_divergence(chain_nn,chain_fem, bins):
    hist_nn, bin_edges = np.histogram(chain_nn, bins=bins)
    hist_fem, _ = np.histogram(chain_fem, bins=bin_edges)

    # Convert to probabilities (PMF)
    p = hist_nn / np.sum(hist_nn)
    q = hist_fem / np.sum(hist_fem)

    # Avoid zero probabilities
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    # KL divergences
    kl_pq = entropy(p, q)  # KL(P || Q)
    kl_qp = entropy(q, p)  # KL(Q || P)

    return kl_pq,kl_qp

def total_kl_divergence(chain_nn,chain_fem, bins=50):
    kl = 0 
    for i in range(chain_nn.shape[-1]):
        kl_pq,_ = discrete_kl_divergence(chain_nn[:,i],chain_fem[:,i], bins)
        kl += kl_pq
    
    return kl

def discrete_2wasserstein(chain_nn,chain_fem,m):
    p = np.linspace(0, 1, m)
    # Quantiles
    Q1 = np.quantile(chain_nn, p)
    Q2 = np.quantile(chain_fem, p)

    # Squared Wasserstein distance
    W = np.trapz((np.abs(Q1 - Q2))**2, p)

    return W

def total_2wasserstein(chain_nn,chain_fem,m=10**5):
    w = 0
    for i in range(chain_nn.shape[-1]):
        w += discrete_2wasserstein(chain_nn[:,i],chain_fem[:,i],m=m)
    return w

def discrete_W1(chain_nn,chain_fem):
    n = chain_fem.shape[0]
    w1 = 0
    for i in range(chain_nn.shape[-1]):
        samples_surrogate = chain_nn[:,i]
        samples_fine = chain_fem[:,i]

    #     p = np.linspace(0, 1, 10**3)
    # # Quantiles
    #     Q1 = np.quantile(samples_surrogate, p)
    #     Q2 = np.quantile(samples_fine, p)

        # Squared Wasserstein distance
        #w1 = np.trapz((np.abs(Q1 - Q2)), p)

        samples_surrogate,samples_fine = np.sort(samples_surrogate), np.sort(samples_fine)

        w1 = np.abs(samples_surrogate-samples_fine).sum()/n
    return w1


def error_norm_mean(surrogate, fine_model_eval,observation_locations,samples,device,gp=False):
    total_error = 0
    for fine_eval,theta in zip(fine_model_eval,samples):
        fine_pred = torch.tensor(fine_eval,device=device)

        if gp:
            surrogate_pred = surrogate.prediction(theta.reshape(1,-1),var=False)
            
        else:
            data = torch.cat([observation_locations, theta.repeat(observation_locations.size(0), 1)], dim=1).float()
            surrogate_pred = surrogate.u(data.float()).detach()

        error = (torch.linalg.norm((fine_pred.reshape(-1,1)-surrogate_pred.reshape(-1,1)),ord=2)**2).item()
        total_error += error

    return np.sqrt(total_error / samples.shape[0])

def error_norm_marginal(surrogate, fine_model_eval,observation_locations,samples, device, gp=False):
    total_error = 0
    for fine_eval,theta in zip(fine_model_eval,samples):
        fine_pred = torch.tensor(fine_eval, device=device)

        if gp:
            mean_surg, var_surg = surrogate.prediction(theta.reshape(1,-1),var=True)
            var_surg = torch.diag(var_surg)

        else:
            data = torch.cat([observation_locations, theta.repeat(observation_locations.size(0), 1)], dim=1).float()
            mean_surg, var_surg = surrogate(data.float())
            mean_surg = mean_surg.view(-1, 1).detach()
            var_surg = var_surg[:, :, 0].view(-1, 1).detach()
        
        error_mean = torch.linalg.norm((fine_pred.reshape(-1,1)-mean_surg.reshape(-1,1)),ord=2)**2
        error_var = torch.sum(var_surg)
        error = torch.sqrt(error_mean + error_var)
        total_error += error.item()

    return total_error / samples.shape[0]
