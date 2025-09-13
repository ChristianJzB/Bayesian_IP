import time
import wandb

import torch
from torch.utils.data import Dataset,DataLoader

import numpy as np
from scipy.stats import qmc

from .FEM2d_Solver import Elliptic2D_FEM
from .elliptic2d import Elliptic2D

def samples_param(size, nparam, min_val=-1, max_val=1,seed = 65647437836358831880808032086803839626):
    """Sample parameters uniformly from a specified range."""
    rng = np.random.default_rng(seed)
    return rng.uniform(min_val, max_val, size=(size, nparam))

class dGDataset(Dataset):
    def __init__(self, size,param = None,nparam = 2):
        self.nparam = nparam
        self.size = size

        self.data_int, self.left_bc, self.right_bc,self.down_bc,self.up_bc = self.generate_data(size, param)

    def generate_data(self, size, param = None, seed = 65647437836358831880808032086803839626):
        #x = lhs(1, size).reshape(-1, 1)  # Latin Hypercube Sampling for x
        sampler = qmc.LatinHypercube(d=2, seed=seed)
        x = sampler.random(n=size)

        if param is None:
            param = samples_param(size=size, nparam=self.nparam)
        else:
            param = param[:size,:]

        x_tensor = torch.Tensor(x)
        x_, y_  = x_tensor[:,0].reshape(-1,1),x_tensor[:,1].reshape(-1,1)
        param_tensor = torch.Tensor(param)

        data_int = torch.cat([x_tensor, param_tensor], axis=1).float()
        left_bc = torch.cat([torch.zeros_like(x_).float(),y_, param_tensor], axis=1).float()
        right_bc = torch.cat([torch.ones_like(x_).float(),y_, param_tensor], axis=1).float()
        down_bc = torch.cat([x_,torch.zeros_like(y_).float(), param_tensor], axis=1).float()
        up_bc = torch.cat([x_,torch.ones_like(y_).float(), param_tensor], axis=1).float()

        return data_int, left_bc, right_bc,down_bc,up_bc

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data_int[index], self.left_bc[index], self.right_bc[index],self.down_bc[index],self.up_bc[index]

def generate_data(size, nparam,param = None, seed = 65647437836358831880808032086803839626):
    #x = lhs(1, size).reshape(-1, 1)  # Latin Hypercube Sampling for x
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    x = sampler.random(n=size)

    if param is None:
        param = samples_param(size=size, nparam=nparam)
    else:
        param = param[:size,:]

    x_tensor = torch.Tensor(x)
    x_, y_  = x_tensor[:,0].reshape(-1,1),x_tensor[:,1].reshape(-1,1)
    param_tensor = torch.Tensor(param)

    data_int = torch.cat([x_tensor, param_tensor], axis=1).float()
    left_bc = torch.cat([torch.zeros_like(x_).float(),y_, param_tensor], axis=1).float()
    right_bc = torch.cat([torch.ones_like(x_).float(),y_, param_tensor], axis=1).float()
    down_bc = torch.cat([x_,torch.zeros_like(y_).float(), param_tensor], axis=1).float()
    up_bc = torch.cat([x_,torch.ones_like(y_).float(), param_tensor], axis=1).float()

    return data_int, left_bc, right_bc,down_bc,up_bc
 


def generate_test_data(size,param = None,ncells=50, nparam=2):
    """
    Generate test data using the FEM solver for a specified number of samples and parameters.
    """
    N = 50
    xgrid = np.linspace(0,1,N)
    X, Y = np.meshgrid(xgrid, xgrid)

    x_test = np.column_stack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel())))

    # Sample parameters
    if param is None:
        test_samples_param = samples_param(size, nparam=nparam)
    else:
        test_samples_param = param[:size,:]

    solver = Elliptic2D_FEM(np.zeros(nparam),ncells=ncells)

    # Preallocate array for test data
    test_data = np.zeros((size, N*N))

    # Loop through and solve for each parameter set
    for i in range(size):
        # Solve the FEM problem
        solver.theta = test_samples_param[i, :]  # Update the parameter vector
        solver.solve()
        solution = solver.evaluate_at_points(x_test)
        test_data[i, :] = solution.reshape(-1)

    return x_test[:,:-1], test_samples_param, test_data


def train_elliptic2d(config, device):
    """Train the PINN using the Adam optimizer with early stopping, computing test loss after each epoch."""
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    torch.manual_seed(config.seed)
    start_scheduler = int(config.epochs * config.start_scheduler )

    data_parameters = samples_param(config.nn_samples*2, nparam=config.KL_expansion)
    
    param_train, param_test = data_parameters[:config.nn_samples,:],  data_parameters[config.nn_samples:,:]

    dataset = dGDataset(size = config.nn_samples, param=param_train)

    x_val,param_val, sol_val = generate_test_data(config.nn_samples,param =param_test,nparam=config.KL_expansion)

    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle=False)

    dg_elliptic = Elliptic2D(config=config, device=device)

    loss_fn = torch.nn.MSELoss(reduction ='mean')
    optimizer = torch.optim.Adam(dg_elliptic.model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)  # Exponential decay scheduler

    for epoch in range(config.epochs):
        epoch_train_loss = 0.0  # Accumulate loss over all batches for this epoch
        update_weights = (epoch % config.weights_update == 0)

        for data_int, left_bc, right_bc,down_bc,up_bc in dataloader:
            data_int, left_bc, right_bc,down_bc,up_bc = data_int.to(device), left_bc.to(device), right_bc.to(device),down_bc.to(device),up_bc.to(device)

            optimizer.zero_grad()

            start_time = time.time()
            total_loss,losses = dg_elliptic.total_loss(data_int, left_bc, right_bc,down_bc,up_bc, 
                                                       loss_fn,update_weights=update_weights)
            loss_computation_time = time.time() - start_time  # Time taken for loss computation

            total_loss.backward()
            optimizer.step()
            
            # Accumulate the batch loss into the epoch loss
            epoch_train_loss += total_loss.item()

        # Calculate the average loss for the epoch
        epoch_train_loss /= len(dataloader)

        # Compute the test loss at the end of the epoch
        if (epoch % 1000 == 0) and (epoch != 0):
            test_loss_current = compute_mean_error(dg_elliptic.model, param_val, x_val, sol_val,device=device)
            wandb.log({"test_loss":test_loss_current})
       

        # Scheduler step
        if epoch >= start_scheduler and (epoch - start_scheduler) % config.scheduler_step == 0:
            scheduler.step()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "loss_computation_time": loss_computation_time,
            "learning_rate": scheduler.get_last_lr()[0],
            **{f"loss_{key}": value.item() for key, value in losses.items()},
        })
        # Save the model checkpoint
        if (epoch % 1000 == 0) and (epoch != 0):
            torch.save(dg_elliptic, f"./Elliptic2D/models/{wandb_config.name}.pth")

    # Save final model
    torch.save(dg_elliptic, f"./Elliptic2D/models/{wandb_config.name}.pth")
    wandb.save(f"./Elliptic2D/models/{wandb_config.name}.pth")

    # Finish W&B run
    wandb.finish()

    return dg_elliptic


def compute_mean_error(model, parameters_test, t, y_numerical,device):
    """
    Compute the mean error between the numerical solution and model predictions.
    """
    error = []
    
    for n, pr1 in enumerate(parameters_test):
        # Generate test data
        
        data_test = np.hstack((t, np.ones((t.shape[0],pr1.shape[0])) * pr1))
        
        # Get the numerical solution for this test case
        numerical_sol = y_numerical[n, :]

        # Predict using the model
        u_pred = model(torch.tensor(data_test,device=device).float()).detach().cpu().numpy()

        # Reshape the numerical solution to match the prediction shape
        numerical_sol = numerical_sol.reshape(u_pred.shape)

        # Calculate the error
        error.append(np.linalg.norm(numerical_sol - u_pred, ord=2) / np.linalg.norm(numerical_sol, ord=2))

    return np.mean(error)