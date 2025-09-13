import torch
import numpy as np
from scipy.optimize import minimize
from Base.utilities import RootFinder

class KernelFunction:
    def __init__(self, device):
        self.device = device

    def euclidean_distance(self, X, Y):
        return torch.cdist(X, Y)
    
    def d_euclidean_distance(self, X, Y):
        # X: (n, 1), Y: (m, 1)
        d = self.euclidean_distance(X, Y)  # shape (n, m)
        delta = X - Y.T                    # shape (n, m)

        # Avoid dividing by zero — we'll set those to 0
        zero_mask = (d == 0)
        dr_dx1 = torch.where(zero_mask, torch.zeros_like(d), delta / d)
        
        return dr_dx1, -dr_dx1
    
    def kernel(self,derivative_order=0):
        raise NotImplementedError("covariance must be implemented in a subclass.")
    
    def covariance(self, X,Y=None, derivative_order=0):
        Y = X if Y is None else Y.to(dtype=X.dtype)
        d = self.euclidean_distance(X, Y)
        return self.kernel(d, derivative_order=derivative_order)

class MaternKernel(KernelFunction):
    def __init__(self, sigma=1.0, l=1.0, device='cpu'):
        super().__init__(device)
        self._sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
        self._l = torch.tensor(l, dtype=torch.float64, device=device)
        self.sqrt5 = torch.sqrt(torch.tensor(5.0, dtype=torch.float64, device=device))

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        if not isinstance(new_sigma, torch.Tensor):
            new_sigma = torch.tensor(new_sigma, dtype=torch.float64, device=self.device, requires_grad=False)
        else:
            new_sigma = new_sigma.to(dtype=torch.float64, device=self.device)
        self._sigma = new_sigma

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, new_l):
        if not isinstance(new_l, torch.Tensor):
            new_l = torch.tensor(new_l, dtype=torch.float64, device=self.device, requires_grad=False)
        else:
            new_l = new_l.to(dtype=torch.float64, device=self.device)
        self._l = new_l

    def matern52_kernel(self, r):
        term1 = 1 + self.sqrt5 * r / self.l + 5 * r**2 / (3 * self.l**2)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def d_matern52_kernel(self, r):
        term1 = -(5 / 3) * (r / self.l**2) - (self.sqrt5 * 5 * r**2) / (3 * self.l**3)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)

    def dd_matern52_kernel(self, r):
        term1 = -5 / (3 * self.l**2) - (5 / 3) * (r * self.sqrt5 / self.l**3) + (25 * r**2) / (3 * self.l**4)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def ddd_matern52_kernel(self, r):
        #term1 = 25 / (3 * self.l**4) + (75 * r)/ (3 * self.l**4) -  (25 * self.sqrt5 * r**2)/ (3 * self.l**5)
        term1 = (75 * r)/ (3 * self.l**4) -  (25 * self.sqrt5 * r**2)/ (3 * self.l**5)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)
    
    def dddd_matern52_kernel(self, r):
        #term1 = 50 / (3 * self.l**4) - (25 * self.sqrt5)/ (3 * self.l**5) + (r**2 * 5**3) / (3*self.l**6)
        term1 = 75 / (3 * self.l**4) - (125 * self.sqrt5*r)/ (3 * self.l**5) + (r**2 * 125) / (3*self.l**6)
        return self.sigma**2 * term1 * torch.exp(-self.sqrt5 * r / self.l)

    def kernel(self, r, derivative_order=0):
        if derivative_order == 0:
            return self.matern52_kernel(r)
        elif derivative_order == 1:
            return self.d_matern52_kernel(r)
        elif derivative_order == 2:
            return self.dd_matern52_kernel(r)
        elif derivative_order == 3:
            return self.ddd_matern52_kernel(r)
        elif derivative_order == 4:
            return self.dddd_matern52_kernel(r)
        else:
            raise ValueError(f"Unsupported derivative order: {derivative_order}")

class SquaredExponential(KernelFunction):
    def __init__(self, sigma=1.0, l=1.0, device='cpu'):
        super().__init__(device)
        self._sigma = torch.tensor(sigma, dtype=torch.float64, device=device)
        self._l = torch.tensor(l, dtype=torch.float64, device=device)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma):
        if not isinstance(new_sigma, torch.Tensor):
            new_sigma = torch.tensor(new_sigma, dtype=torch.float64, device=self.device, requires_grad=False)
        else:
            new_sigma = new_sigma.to(dtype=torch.float64, device=self.device)
        self._sigma = new_sigma

    @property
    def l(self):
        return self._l

    @l.setter
    def l(self, new_l):
        if not isinstance(new_l, torch.Tensor):
            new_l = torch.tensor(new_l, dtype=torch.float64, device=self.device, requires_grad=False)
        else:
            new_l = new_l.to(dtype=torch.float64, device=self.device)
        self._l = new_l

    def squared_exponential_cov(self, r):
        return (self.sigma**2) * torch.exp(-0.5 * (r / self.l) ** 2)

    def d_squared_exponential_cov(self, r):
        return -((self.sigma / self.l) ** 2) * r * torch.exp(-0.5 * (r / self.l) ** 2)

    def dd_squared_exponential_cov(self, r):
            return (-1/self.l**2 + r**2/self.l**4 )*self.squared_exponential_cov(r)
    
    def ddd_squared_exponential_cov(self, r):
            return (3*r / self.l**4 - r**3 / self.l**6)*self.squared_exponential_cov(r)
    
    def dddd_squared_exponential_cov(self, r):
            return (3 / self.l**4 - 6*r**2 /self.l**6 + r**4 / self.l**8)*self.squared_exponential_cov(r)

    def kernel(self, r, derivative_order=0):
        if derivative_order == 0:
            return self.squared_exponential_cov(r)
        elif derivative_order == 1:
            return self.d_squared_exponential_cov(r)
        elif derivative_order == 2:
            return self.dd_squared_exponential_cov(r)
        elif derivative_order == 3:
            return self.ddd_squared_exponential_cov(r)
        elif derivative_order == 4:
            return self.dddd_squared_exponential_cov(r)
        else:
            raise ValueError(f"Unsupported derivative order: {derivative_order}")


class EllipticPIGP:
    def __init__(self, data_training, reg_matrix=1e-6, lam=1/4,sigma_l_parameters=(1, 1), sigma_l_spatial=(1, 1),device = "cpu"):

        self.device = device
        # Training data
        self.parameters_data = torch.tensor(data_training["parameters_data"], dtype=torch.float64)
        self.solutions_data = torch.tensor(data_training["solutions_data"], dtype=torch.float64)
        self.x_sol_data = torch.tensor(data_training["x_solutions_data"], dtype=torch.float64)

        self.x_bc = torch.tensor(data_training["x_bc"], dtype=torch.float64)
        self.y_bc = torch.tensor(data_training["y_bc"], dtype=torch.float64)

        self.source_func_x = torch.tensor(data_training["source_func_x"], dtype=torch.float64).view(-1, 1)
        self.source_func_f_x = torch.tensor(data_training["source_func_f_x"], dtype=torch.float64)

        self.n_parameter_obs = self.parameters_data.shape[0]
        self.parameter_dim = self.parameters_data.shape[-1]

        # Roots for KL
        self.finder = RootFinder(lam, self.parameter_dim)
        self.roots = torch.tensor(self.finder.find_roots(), dtype=torch.float64)

        self.kernel_parameter = SquaredExponential(*sigma_l_parameters,device=self.device)
        self.kernel_spatial = MaternKernel(*sigma_l_spatial,device=self.device)
        
        self.reg_matrix = reg_matrix

        self.g_trained = self.g_training()
        self.kuf = self.kernel_uf(self.parameters_data,self.x_sol_data)
        self.cov_matrix, self.kuu, self.kug = self.informed_kernel(self.parameters_data, self.parameters_data)
        self.invk_g = self.kernel_inverse(self.cov_matrix, self.g_trained)

    @property
    def A(self):
        r = self.roots
        return torch.sqrt(1 / ((1/8)*(5 + (r / 2)**2) + 
                            (torch.sin(2*r) / (4*r)) * ((r / 4)**2 - 1) - (torch.cos(2*r)/8)))
    @property
    def an(self):
        return torch.sqrt(8 / (self.roots**2 + 16))

    def exp_kl_eval(self, theta, x):
        basis = self.A[:, None] * self.an[:, None] * (torch.sin(self.roots[:, None] * x) +
                    (self.roots[:, None] / 4) * torch.cos(self.roots[:, None] * x))
        result = theta @ basis
        return torch.exp(result)

    def grad_kl_eval(self, theta, x):
        basis = self.A[:, None] * self.an[:, None] * self.roots[:, None] * (
            torch.cos(self.roots[:, None] * x) - (self.roots[:, None] / 4) * torch.sin(self.roots[:, None] * x))
        return theta @ basis


    def kernel_uf(self, theta, x):
        exp_kl = self.exp_kl_eval(theta, self.source_func_x.T)
        grad_kl = self.grad_kl_eval(theta, self.source_func_x.T)

        _,dr_dx = self.kernel_spatial.d_euclidean_distance(x, self.source_func_x)
        dk_dr = self.kernel_spatial.covariance(x,self.source_func_x, derivative_order=1)
        ddk_ddr = self.kernel_spatial.covariance(x,self.source_func_x, derivative_order=2)
    
        # Reshape for broadcasting
        exp_kl = exp_kl.unsqueeze(1)           # [B, 1, S]
        grad_kl = grad_kl.unsqueeze(1)         # [B, 1, S]
        dr_dx   = dr_dx.unsqueeze(0)           # [1, N, S]
        dk_dr   = dk_dr.unsqueeze(0)           # [1, N, S]
        ddk_ddr = ddk_ddr.unsqueeze(0)         # [1, N, S]

        # Compute [B, N, S] via broadcasting
        pi_gp = grad_kl * dk_dr * dr_dx + ddk_ddr 
        return -pi_gp * exp_kl  # [B, N, S]

    def kernel_ff(self, theta1, theta2):

        exp_kl1 = self.exp_kl_eval(theta1, self.source_func_x.T).view(-1, 1)
        grad_kl1 = self.grad_kl_eval(theta1, self.source_func_x.T).view(-1, 1)
        exp_kl2 = self.exp_kl_eval(theta2, self.source_func_x.T).view(-1, 1)
        grad_kl2 = self.grad_kl_eval(theta2, self.source_func_x.T).view(-1, 1)

        dr_dx1,dr_dx2 = self.kernel_spatial.d_euclidean_distance(self.source_func_x, self.source_func_x)
        ddk = self.kernel_spatial.covariance(self.source_func_x,self.source_func_x, derivative_order=2)
        dddk = self.kernel_spatial.covariance(self.source_func_x,self.source_func_x, derivative_order=3)
        ddddk = self.kernel_spatial.covariance(self.source_func_x,self.source_func_x, derivative_order=4)

        llk = (-grad_kl1*grad_kl2.T*ddk + grad_kl1*dddk*dr_dx1 - grad_kl2.T*dddk*dr_dx1 + ddddk)*exp_kl1*exp_kl2.T
        return llk 

    def block_matrix_builder_ff(self, th1, th2):
        nth1, nth2 = th1.shape[0], th2.shape[0]
        blocks = []

        for i in range(nth1):
            row_blocks = []
            for j in range(nth2):
                theta1 = th1[i, :]
                theta2 = th2[j, :]
                cov_scalar = self.kernel_parameter.covariance(theta1.view(1, -1), theta2.view(1, -1))
                pde_matrix = self.kernel_ff(theta1.unsqueeze(0), theta2.unsqueeze(0))
                row_blocks.append(cov_scalar * pde_matrix)
            blocks.append(torch.cat(row_blocks, dim=1))
        return torch.cat(blocks, dim=0)
    

    def block_matrix_builder_uf(self, th1, th2, x = None):
        cov_matrix = self.kernel_parameter.covariance(th1, th2)  # [nth1, nth2]

        if x is None:
            pde_matrices = self.kuf # [nth2, N, S]
            x = self.x_sol_data
        else:
            pde_matrices = self.kernel_uf(th2, x)

        # Broadcast cov scalars over PDEs
        # [nth1, nth2, 1, 1] * [1, nth2, N, S] → [nth1, nth2, N, S]
        blocks = cov_matrix.unsqueeze(-1).unsqueeze(-1) * pde_matrices.unsqueeze(0)

        # Stack the final block matrix
        result = torch.cat(
            [blocks[i].transpose(0, 1).reshape(x.shape[0], -1) for i in range(th1.shape[0])],
            dim=0
        )  # shape: [nth1 * N, nth2 * S]
        return result

    def informed_kernel(self, theta1, theta2):
        kernel_param = self.kernel_parameter.covariance(theta1, theta2)

        kuu = self.kernel_spatial.covariance(self.x_sol_data, self.x_sol_data)
        Kuu = torch.kron(kernel_param, kuu)

        kug = self.kernel_spatial.covariance(self.x_sol_data, self.x_bc)
        Kug = torch.kron(kernel_param, kug)

        Kuf = self.block_matrix_builder_uf(theta1, theta2)

        kgg = self.kernel_spatial.covariance(self.x_bc, self.x_bc)
        Kgg = torch.kron(kernel_param, kgg)

        Kgf = self.block_matrix_builder_uf(theta1, theta2, self.x_bc)
        Kff = self.block_matrix_builder_ff(theta1, theta2)

        top = torch.cat([Kuu, Kug, Kuf], dim=1)
        middle = torch.cat([Kug.T, Kgg, Kgf], dim=1)
        bottom = torch.cat([Kuf.T, Kgf.T, Kff], dim=1)
        cov = torch.cat([top, middle, bottom], dim=0)

        return cov + self.reg_matrix * torch.eye(cov.shape[0], dtype=torch.float64),kuu,kug

    def g_training(self):
        y_bc_n = self.y_bc.repeat((self.n_parameter_obs,1))
        f_source_n = self.source_func_f_x.repeat((self.n_parameter_obs,1))
        return torch.cat([self.solutions_data.view(-1, 1), y_bc_n, f_source_n])

    def kernel_inverse(self, cov_matrix, Y):
        return torch.linalg.solve(cov_matrix, Y)

    def marginal_likelihood(self, sigma_spatial, l_spatial, sigma_param, l_param):
        # Update kernel parameters
        self.kernel_spatial.sigma, self.kernel_spatial.l = sigma_spatial, l_spatial
        self.kernel_parameter.sigma, self.kernel_parameter.l = sigma_param, l_param

        # Recompute the covariance matrix
        self.kuf = self.kernel_uf(self.parameters_data,self.x_sol_data)
        self.cov_matrix, _, _ = self.informed_kernel(self.parameters_data, self.parameters_data)
        
        # Compute Cholesky
        L = torch.linalg.cholesky(self.cov_matrix)

        # Solve K^{-1}y
        y = self.g_trained
        alpha =self.kernel_inverse(self.cov_matrix,y)

        # Log determinant
        logdet_K = 2.0 * torch.sum(torch.log(torch.diag(L)))
        return -0.5 * ((y.T @ alpha).squeeze() + logdet_K + y.shape[0] * np.log(2 * np.pi))


    def optimize_mll(self, lr=1e-2):
    # Start from log of current parameters
        log_sigma_spatial = self.kernel_spatial.sigma.detach().log().clone().requires_grad_()
        log_l_spatial = self.kernel_spatial.l.detach().log().clone().requires_grad_()
        log_sigma_param = self.kernel_parameter.sigma.detach().log().clone().requires_grad_()
        log_l_param = self.kernel_parameter.l.detach().log().clone().requires_grad_()

        params = [log_sigma_spatial, log_l_spatial, log_sigma_param, log_l_param]
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=100, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()

            # Exponentiate to ensure positivity
            sigma_spatial = log_sigma_spatial.exp()
            l_spatial = log_l_spatial.exp()
            sigma_param = log_sigma_param.exp()
            l_param = log_l_param.exp()

            nll = -self.marginal_likelihood(sigma_spatial, l_spatial, sigma_param, l_param)
            nll.backward()
            return nll

        optimizer.step(closure)

        # Update final values after optimization
        self.kernel_spatial.sigma = log_sigma_spatial.exp().detach()
        self.kernel_spatial.l = log_l_spatial.exp().detach()
        self.kernel_parameter.sigma = log_sigma_param.exp().detach()
        self.kernel_parameter.l = log_l_param.exp().detach()

        print("Optimized parameters:")
        print(f"(sigma_spatial,l_spatial) = ({self.kernel_spatial.sigma},{self.kernel_spatial.l})")
        print(f"(sigma_parameter,l_parameter) = ({self.kernel_parameter.sigma},{self.kernel_parameter.l})")
        
        self.kuf = self.kernel_uf(self.parameters_data,self.x_sol_data)
        self.cov_matrix, self.kuu,self.kug = self.informed_kernel(self.parameters_data, self.parameters_data)
        self.invk_g = self.kernel_inverse(self.cov_matrix, self.g_trained)


    def marginal_val(self, theta_test, x_test=None):
        kernel_param = self.kernel_parameter.covariance(theta_test, self.parameters_data)

        if x_test is None:
            x_test = self.x_sol_data
            Kuf = self.block_matrix_builder_uf(theta_test, self.parameters_data)
            kuu = self.kuu
            kug = self.kug
        else: 
            Kuf = self.block_matrix_builder_uf(theta_test, self.parameters_data,x_test)

            kuu = self.kernel_spatial.covariance(x_test, self.x_sol_data)
            
            kug = self.kernel_spatial.covariance(x_test, self.x_bc)

        Kuu = torch.kron(kernel_param, kuu)

        Kug = torch.kron(kernel_param, kug)
        
        return torch.cat([Kuu, Kug, Kuf], dim=1)

    def prediction(self, theta_test, x_test=None, var=True):
        matrix_test = self.marginal_val(theta_test, x_test)
        marginal_mean = matrix_test @ self.invk_g

        if var:
            kernel_spatial = self.kuu if x_test is None else  self.kernel_spatial.covariance(x_test, x_test)
            kernel_param = self.kernel_parameter.covariance(theta_test, theta_test)
            kinv_y = self.kernel_inverse(self.cov_matrix, matrix_test.T)
            cov = torch.kron(kernel_param, kernel_spatial) - matrix_test @ kinv_y
            return marginal_mean, cov
        return marginal_mean














class PIGP:
    def __init__(self, data_training, reg_matrix=1e-6,sigma_l_parameters=(1, 1), sigma_l_spatial=(1, 1),device = "cpu"):

        self.device = device
        # Training data
        self.parameters_data = torch.tensor(data_training["parameters_data"], dtype=torch.float64,device=self.device)
        self.solutions_data = torch.tensor(data_training["solutions_data"], dtype=torch.float64,device=self.device)
        self.x_sol_data = torch.tensor(data_training["x_solutions_data"], dtype=torch.float64,device=self.device)

        self.x_bc = torch.tensor(data_training["x_bc"], dtype=torch.float64,device=self.device)
        self.y_bc = torch.tensor(data_training["y_bc"], dtype=torch.float64,device=self.device)

        self.source_func_x = torch.tensor(data_training["source_func_x"], dtype=torch.float64,device=self.device)
        self.source_func_f_x = torch.tensor(data_training["source_func_f_x"], dtype=torch.float64,device=self.device)

        self.n_parameter_obs = self.parameters_data.shape[0]
        self.parameter_dim = self.parameters_data.shape[-1]

        self.kernel_parameter = SquaredExponential(*sigma_l_parameters,device=self.device)
        self.kernel_spatial = MaternKernel(*sigma_l_spatial,device=self.device)
        
        self.reg_matrix = reg_matrix

    def train_gp(self):
          self.g_trained = self.g_training()
          self.kuf = self.kernel_uf(self.parameters_data,self.x_sol_data)
          self.cov_matrix, self.kuu, self.kug = self.informed_kernel(self.parameters_data, self.parameters_data)
          self.invk_g = self.kernel_inverse(self.cov_matrix, self.g_trained)

    def exp_kl_eval(self, theta, x):
        """Define the likelihood function for the coarse model. Must be overridden."""
        raise NotImplementedError("log_likelihood must be implemented in a subclass.")
     
    def grad_kl_eval(self, theta, x):
        """Define the likelihood function for the coarse model. Must be overridden."""
        raise NotImplementedError("log_likelihood must be implemented in a subclass.")
    
    def kernel_uf(self, theta, x):
        exp_kl = self.exp_kl_eval(theta, self.source_func_x.T)
        grad_kl = self.grad_kl_eval(theta, self.source_func_x.T)

        _,dr_dx = self.kernel_spatial.d_euclidean_distance(x, self.source_func_x)
        dk_dr = self.kernel_spatial.covariance(x,self.source_func_x, derivative_order=1)
        ddk_ddr = self.kernel_spatial.covariance(x,self.source_func_x, derivative_order=2)
    
        # Reshape for broadcasting
        exp_kl = exp_kl.unsqueeze(1)           # [B, 1, S]
        grad_kl = grad_kl.unsqueeze(1)         # [B, 1, S]
        dr_dx   = dr_dx.unsqueeze(0)           # [1, N, S]
        dk_dr   = dk_dr.unsqueeze(0)           # [1, N, S]
        ddk_ddr = ddk_ddr.unsqueeze(0)         # [1, N, S]

        # Compute [B, N, S] via broadcasting
        pi_gp = grad_kl * dk_dr * dr_dx + ddk_ddr 
        return -pi_gp * exp_kl  # [B, N, S]
    
    def kernel_ff(self, theta1, theta2):

        exp_kl1 = self.exp_kl_eval(theta1, self.source_func_x.T).view(-1, 1)
        grad_kl1 = self.grad_kl_eval(theta1, self.source_func_x.T).view(-1, 1)
        exp_kl2 = self.exp_kl_eval(theta2, self.source_func_x.T).view(-1, 1)
        grad_kl2 = self.grad_kl_eval(theta2, self.source_func_x.T).view(-1, 1)

        dr_dx1,dr_dx2 = self.kernel_spatial.d_euclidean_distance(self.source_func_x, self.source_func_x)
        ddk = self.kernel_spatial.covariance(self.source_func_x,self.source_func_x, derivative_order=2)
        dddk = self.kernel_spatial.covariance(self.source_func_x,self.source_func_x, derivative_order=3)
        ddddk = self.kernel_spatial.covariance(self.source_func_x,self.source_func_x, derivative_order=4)

        llk = (-grad_kl1*grad_kl2.T*ddk + (grad_kl1- grad_kl2.T)*dddk*dr_dx1 + ddddk)*exp_kl1*exp_kl2.T
        return llk 
    
    def block_matrix_builder_ff(self, th1, th2):
        nth1, nth2 = th1.shape[0], th2.shape[0]
        blocks = []

        for i in range(nth1):
            row_blocks = []
            for j in range(nth2):
                theta1 = th1[i, :]
                theta2 = th2[j, :]
                cov_scalar = self.kernel_parameter.covariance(theta1.view(1, -1), theta2.view(1, -1))
                pde_matrix = self.kernel_ff(theta1.unsqueeze(0), theta2.unsqueeze(0))
                row_blocks.append(cov_scalar * pde_matrix)
            blocks.append(torch.cat(row_blocks, dim=1))
        return torch.cat(blocks, dim=0)
    
    def block_matrix_builder_uf(self, th1, th2, x = None):
        cov_matrix = self.kernel_parameter.covariance(th1, th2)  # [nth1, nth2]

        if x is None:
            pde_matrices = self.kuf # [nth2, N, S]
            x = self.x_sol_data
        else:
            pde_matrices = self.kernel_uf(th2, x)

        # Broadcast cov scalars over PDEs
        # [nth1, nth2, 1, 1] * [1, nth2, N, S] → [nth1, nth2, N, S]
        blocks = cov_matrix.unsqueeze(-1).unsqueeze(-1) * pde_matrices.unsqueeze(0)

        # Stack the final block matrix
        result = torch.cat(
            [blocks[i].transpose(0, 1).reshape(x.shape[0], -1) for i in range(th1.shape[0])],
            dim=0
        )  # shape: [nth1 * N, nth2 * S]
        return result
    
    def informed_kernel(self, theta1, theta2):
        kernel_param = self.kernel_parameter.covariance(theta1, theta2)

        kuu = self.kernel_spatial.covariance(self.x_sol_data, self.x_sol_data)
        Kuu = torch.kron(kernel_param, kuu)

        kug = self.kernel_spatial.covariance(self.x_sol_data, self.x_bc)
        Kug = torch.kron(kernel_param, kug)

        Kuf = self.block_matrix_builder_uf(theta1, theta2)

        kgg = self.kernel_spatial.covariance(self.x_bc, self.x_bc)
        Kgg = torch.kron(kernel_param, kgg)

        Kgf = self.block_matrix_builder_uf(theta1, theta2, self.x_bc)
        Kff = self.block_matrix_builder_ff(theta1, theta2)

        top = torch.cat([Kuu, Kug, Kuf], dim=1)
        middle = torch.cat([Kug.T, Kgg, Kgf], dim=1)
        bottom = torch.cat([Kuf.T, Kgf.T, Kff], dim=1)
        cov = torch.cat([top, middle, bottom], dim=0)

        return cov + self.reg_matrix * torch.eye(cov.shape[0], dtype=torch.float64, device=self.device),kuu,kug
    
    def g_training(self):
        y_bc_n = self.y_bc.repeat((self.n_parameter_obs,1))
        f_source_n = self.source_func_f_x.repeat((self.n_parameter_obs,1))
        return torch.cat([self.solutions_data.view(-1, 1), y_bc_n, f_source_n])

    def kernel_inverse(self, cov_matrix, Y):
        return torch.linalg.solve(cov_matrix, Y)

    def marginal_likelihood(self, sigma_spatial, l_spatial, sigma_param, l_param):
        # Update kernel parameters
        self.kernel_spatial.sigma, self.kernel_spatial.l = sigma_spatial, l_spatial
        self.kernel_parameter.sigma, self.kernel_parameter.l = sigma_param, l_param

        # Recompute the covariance matrix
        self.kuf = self.kernel_uf(self.parameters_data,self.x_sol_data)
        self.cov_matrix, _, _ = self.informed_kernel(self.parameters_data, self.parameters_data)
        
        # Compute Cholesky
        L = torch.linalg.cholesky(self.cov_matrix)

        # Solve K^{-1}y
        y = self.g_trained
        alpha =self.kernel_inverse(self.cov_matrix,y)

        # Log determinant
        logdet_K = 2.0 * torch.sum(torch.log(torch.diag(L)))
        return -0.5 * ((y.T @ alpha).squeeze() + logdet_K + y.shape[0] * np.log(2 * np.pi))
    

    def optimize_mll(self, lr=1e-2):
    # Start from log of current parameters
        log_sigma_spatial = self.kernel_spatial.sigma.detach().log().clone().requires_grad_()
        log_l_spatial = self.kernel_spatial.l.detach().log().clone().requires_grad_()
        log_sigma_param = self.kernel_parameter.sigma.detach().log().clone().requires_grad_()
        log_l_param = self.kernel_parameter.l.detach().log().clone().requires_grad_()

        params = [log_sigma_spatial, log_l_spatial, log_sigma_param, log_l_param]
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=100, line_search_fn='strong_wolfe')

        def closure():
            optimizer.zero_grad()

            # Exponentiate to ensure positivity
            sigma_spatial = log_sigma_spatial.exp()
            l_spatial = log_l_spatial.exp()
            sigma_param = log_sigma_param.exp()
            l_param = log_l_param.exp()

            nll = -self.marginal_likelihood(sigma_spatial, l_spatial, sigma_param, l_param)
            nll.backward()
            return nll

        optimizer.step(closure)

        # Update final values after optimization
        self.kernel_spatial.sigma = log_sigma_spatial.exp().detach()
        self.kernel_spatial.l = log_l_spatial.exp().detach()
        self.kernel_parameter.sigma = log_sigma_param.exp().detach()
        self.kernel_parameter.l = log_l_param.exp().detach()

        print("Optimized parameters:")
        print(f"(sigma_spatial,l_spatial) = ({self.kernel_spatial.sigma},{self.kernel_spatial.l})")
        print(f"(sigma_parameter,l_parameter) = ({self.kernel_parameter.sigma},{self.kernel_parameter.l})")
        self.train_gp()

    def marginal_val(self, theta_test, x_test=None):
        kernel_param = self.kernel_parameter.covariance(theta_test, self.parameters_data)

        if x_test is None:
            x_test = self.x_sol_data
            Kuf = self.block_matrix_builder_uf(theta_test, self.parameters_data)
            kuu = self.kuu
            kug = self.kug
        else: 
            Kuf = self.block_matrix_builder_uf(theta_test, self.parameters_data,x_test)

            kuu = self.kernel_spatial.covariance(x_test, self.x_sol_data)
            
            kug = self.kernel_spatial.covariance(x_test, self.x_bc)

        Kuu = torch.kron(kernel_param, kuu)

        Kug = torch.kron(kernel_param, kug)
        
        return torch.cat([Kuu, Kug, Kuf], dim=1)

    def prediction(self, theta_test, x_test=None, var=True):
        matrix_test = self.marginal_val(theta_test, x_test)
        marginal_mean = matrix_test @ self.invk_g

        if var:
            kernel_spatial = self.kuu if x_test is None else  self.kernel_spatial.covariance(x_test, x_test)
            kernel_param = self.kernel_parameter.covariance(theta_test, theta_test)
            kinv_y = self.kernel_inverse(self.cov_matrix, matrix_test.T)
            cov = torch.kron(kernel_param, kernel_spatial) - matrix_test @ kinv_y
            return marginal_mean, cov
        return marginal_mean
    
class Elliptic1DPIGP(PIGP):
    def __init__(self, data_training, reg_matrix=1e-6, lam=1/4,sigma_l_parameters=(1, 1), sigma_l_spatial=(1, 1),device = "cpu"):
        super(Elliptic1DPIGP, self).__init__(data_training = data_training , reg_matrix=reg_matrix,
                             sigma_l_parameters=sigma_l_parameters, sigma_l_spatial=sigma_l_spatial,device=device)
            
        # Roots for KL
        self.finder = RootFinder(lam, self.parameter_dim)
        self.roots = torch.tensor(self.finder.find_roots(), dtype=torch.float64,device=self.device)
        self.kuf = self.kernel_uf(self.parameters_data,self.x_sol_data)


    @property
    def A(self):
        r = self.roots
        return torch.sqrt(1 / ((1/8)*(5 + (r / 2)**2) + 
                            (torch.sin(2*r) / (4*r)) * ((r / 4)**2 - 1) - (torch.cos(2*r)/8)))
    @property
    def an(self):
        return torch.sqrt(8 / (self.roots**2 + 16))
    
    def exp_kl_eval(self, theta, x):
        basis = self.A[:, None] * self.an[:, None] * (torch.sin(self.roots[:, None] * x) +
                    (self.roots[:, None] / 4) * torch.cos(self.roots[:, None] * x))
        result = theta @ basis
        return torch.exp(result).to(self.device)

    def grad_kl_eval(self, theta, x):
        basis = self.A[:, None] * self.an[:, None] * self.roots[:, None] * (
            torch.cos(self.roots[:, None] * x) - (self.roots[:, None] / 4) * torch.sin(self.roots[:, None] * x))
        res = theta @ basis
        return res.to(self.device)