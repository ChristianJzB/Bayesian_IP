
import torch
import numpy as np
from Base.dg import deepGalerkin
from Base.utilities import RootFinder, compute_seq_pairs

class Elliptic2D(deepGalerkin):
    def __init__(self, config,device, lam = 1/4):
        super().__init__(config,device)
        #self.eigen_pairs = compute_seq_pairs(config.KL_expansion)
        self.KL_expansion = config.KL_expansion
        self.trunc_Nx = int(np.ceil(0.5*config.KL_expansion + 1))
        self.root_finder = RootFinder(lam, self.trunc_Nx)
        self.roots = torch.tensor(self.root_finder.find_roots(),device = device)

    @property
    def A(self):
        r = self.roots
        return torch.sqrt(1 / ((1/8)*(5 + (r / 2)**2) + 
                            (torch.sin(2*r) / (4*r)) * ((r / 4)**2 - 1) - (torch.cos(2*r)/8)))
    @property
    def an(self):
        return torch.sqrt(8 / (self.roots**2 + 16))
    
    @property
    def k_terms_order(self):
        v_ij = torch.outer(self.an, self.an).reshape(-1)
        indices = torch.tensor([(i, j) for i in range(self.trunc_Nx) for j in range(self.trunc_Nx)])
        sort_idx = np.argsort(np.array(v_ij))[::-1].copy()
        #sort_idx =torch.tensor()
        v_ij,indices = v_ij[sort_idx],indices[sort_idx]
        v_ij,indices = v_ij[:self.KL_expansion],indices[:self.KL_expansion]
        kx,ky = indices[:, 0],indices[:, 1]
        A_x, roots_x = self.A[kx],self.roots[kx]
        A_y, roots_y = self.A[ky],self.roots[ky]

        return v_ij,A_x,A_y,roots_x,roots_y
    
    def k_function(self, x):
        x0 = x[:, 0].reshape(-1,1)  # (obs,)
        x1 = x[:, 1].reshape(-1,1)  # (obs,)
        theta = x[:,2:]
        v_ij,A_x,A_y,roots_x,roots_y = self.k_terms_order

        phi_i = A_x * (torch.sin(roots_x * x0) +(roots_x / 4) * torch.cos(roots_x * x0))
        phi_j = A_y * (torch.sin(roots_y * x1) +(roots_y / 4) * torch.cos(roots_y * x1))

        return torch.sum(v_ij*phi_i*phi_j*theta,axis=1)


    
    # def k_function(self, x):
    #     # kx = torch.tensor([p[0] for p in self.eigen_pairs], device=self.device)  # (terms_sum,)
    #     # ky = torch.tensor([p[1] for p in self.eigen_pairs], device=self.device)

    #     # an_kx = self.an[kx]  # (terms_sum,)
    #     # an_ky = self.an[ky]

    #     # roots_kx = self.roots[kx].squeeze(-1)  # (terms_sum,)
    #     # roots_ky = self.roots[ky].squeeze(-1)
    #     # A_kx = self.A[kx].squeeze(-1)
    #     # A_ky = self.A[ky].squeeze(-1)

    #     x0 = x[:, 0].reshape(1,-1)  # (obs,)
    #     x1 = x[:, 1].reshape(1,-1)  # (obs,)
    #     theta = x[:,2:].T
        
    #     # Shape: (terms_sum, obs)
    #     phi_kx = self.A[:,None] * (torch.sin(self.roots[:,None] * x0) +(self.roots[:,None] / 4) * torch.cos(self.roots[:,None] * x0))

    #     phi_ky = self.A[:,None] * ( torch.sin(self.roots[:,None] * x1) + (self.roots[:,None]/ 4) * torch.cos(self.roots[:,None] * x1))

    #     # 4. Basis terms
    #     v_ij = torch.outer(self.an, self.an).reshape(-1)  # (terms_sum,)

    #     # Sort in descending order
    #     sort_idx = torch.argsort(v_ij, descending=True)
    #     v_ij = v_ij[sort_idx][:,None]
    #     print(v_ij)


    #     phi_ij = phi_kx[:, None, :]*phi_ky[None, :, :]  # shape: (n, n, x_dim)
    #     phi_ij = phi_ij.reshape(-1,x.shape[0])

    #     phi_ij = phi_ij[sort_idx,:]

    #     return torch.sum(v_ij[:self.KL_expansion,:]*phi_ij[:self.KL_expansion,:]*theta,dim=0),phi_ij
    
    
    @deepGalerkin.laplace_approx()
    def u(self,x):
        pred = self.model(x)
        return pred.reshape(-1,1)
    
    @deepGalerkin.laplace_approx()
    def elliptic_pde(self, x_interior):
        """ The pytorch autograd version of calculating residual """
        data_domain = x_interior.requires_grad_(True)

        u = self.model(data_domain)

        du = torch.autograd.grad(u, data_domain,grad_outputs=torch.ones_like(u),create_graph=True)[0]

        k = self.k_function(data_domain)
        
        ddu_x = torch.autograd.grad(torch.exp(k).reshape(-1,1)*du[:,0].reshape(-1,1),data_domain, 
            grad_outputs=torch.ones_like(du[:,0].reshape(-1,1)),create_graph=True)[0]
        
        ddu_y = torch.autograd.grad(torch.exp(k).reshape(-1,1)*du[:,1].reshape(-1,1),data_domain, 
            grad_outputs=torch.ones_like(du[:,1].reshape(-1,1)),create_graph=True)[0]            
            
        return ddu_x[:,0].reshape(-1,1) + ddu_y[:,1].reshape(-1,1) + 4*data_domain[:,0].reshape(-1,1)*data_domain[:,1].reshape(-1,1)


    def pde_loss(self,data_interior,loss_fn):
        elliptic_pred = self.elliptic_pde(data_interior)
        zeros = torch.zeros_like(elliptic_pred)
        return loss_fn(elliptic_pred,zeros)
    
    def bc_loss(self,bcl_pts,bcr_pts,bcd_pts,bcu_pts,loss_fn, bc=0):
        u_bcl = self.u(bcl_pts)
        u_bcr = self.u(bcr_pts)
        u_bcu = self.u(bcu_pts)
        u_bcd = self.u(bcd_pts)

        bc_vals = torch.ones_like(u_bcl)* bc

        loss_ubcl = loss_fn(u_bcl, bc_vals)
        loss_ubcr = loss_fn(u_bcr, bc_vals)
        loss_ubcu = loss_fn(u_bcu, bc_vals)
        loss_ubcd = loss_fn(u_bcd, bc_vals)


        return loss_ubcl,loss_ubcr,loss_ubcd,loss_ubcu
    
    def losses(self,data_interior,bcl_pts,bcr_pts,bcd_pts,bcu_pts,loss_fn):
        loss_pde = self.pde_loss(data_interior,loss_fn)
        loss_ubcl,loss_ubcr,loss_ubcd,loss_ubcu = self.bc_loss(bcl_pts,bcr_pts,bcd_pts,bcu_pts,loss_fn)

        losses = {"elliptic":loss_pde, "ubcl":loss_ubcl, "ubcr":loss_ubcr,"ubcd":loss_ubcd,"ubcu":loss_ubcu}
        return losses
    