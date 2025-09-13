import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx import geometry
import numpy as np
from Base.utilities import compute_seq_pairs, RootFinder


    
class Parametric_K:
    def __init__(self, theta, lam =1/4):
        self.theta = np.array(theta)
        self.M = theta.shape[-1]
        self.trunc_Nx = int(np.ceil(0.5*self.M + 1))
        #self.eigen_pairs = compute_seq_pairs(self.M )
        self.finder = RootFinder(lam,self.trunc_Nx)
        self.roots = np.array(self.finder.find_roots())

    @property
    def A(self):
        """Compute the A coefficients."""
        return np.sqrt(1 / ((1/8)*(5 + (self.roots / 2)**2) + 
                            (np.sin(2*self.roots) / (4*self.roots)) * ((self.roots / 4)**2 - 1) - (np.cos(2*self.roots)/8)))
    @property
    def an(self):
        """Compute the an values."""
        return np.sqrt(8 / (self.roots**2 + 16))
    
    @property
    def k_terms_order(self):
        v_ij = np.outer(self.an, self.an).ravel()
        indices = np.array([(i, j) for i in range(self.trunc_Nx) for j in range(self.trunc_Nx)])
        sort_idx = np.argsort(np.array(v_ij))[::-1].copy()
        #sort_idx =torch.tensor()
        v_ij,indices = v_ij[sort_idx],indices[sort_idx]
        v_ij,indices = v_ij[:self.M],indices[:self.M]
        kx,ky = indices[:, 0],indices[:, 1]
        A_x, roots_x = self.A[kx],self.roots[kx]
        A_y, roots_y = self.A[ky],self.roots[ky]

        return v_ij,A_x,A_y,roots_x,roots_y
    
    def eval(self,x,y):
        v_ij,A_x,A_y,roots_x,roots_y = self.k_terms_order
        phi_i = A_x[:,None] * (np.sin(roots_x[:,None] * x[None, :]) + (roots_x[:,None] / 4) * np.cos(roots_x[:,None] * x[None, :]))
        phi_j = A_y[:,None] * (np.sin(roots_y[:,None] * y[None, :]) + (roots_y[:,None] / 4) * np.cos(roots_y[:,None] * y[None, :]))
        v_phi_theta = np.sum(v_ij[:,None]*phi_i*phi_j*self.theta[:,None],axis=0)
        return np.exp(v_phi_theta)
    
    # def eval(self,x,y):
    #     x_dim = x.shape[0]
    #     v_ij = np.outer(self.an, self.an).ravel()
    #     phi_i = self.A[:,None] * (np.sin(self.roots[:,None] * x[None, :]) + (self.roots[:,None] / 4) * np.cos(self.roots[:,None] * x[None, :]))
    #     phi_j = self.A[:,None] * (np.sin(self.roots[:,None] * y[None, :]) + (self.roots[:,None] / 4) * np.cos(self.roots[:,None] * y[None, :]))

    #     sort_idx = np.argsort(v_ij)[::-1]
    #     v_ij = v_ij[sort_idx]

    #     phi_ij= phi_i[:, None, :] * phi_j[None, :, :]  # (100, 100, 16000)
    #     phi_ij = phi_ij.reshape(-1, x_dim)
    #     phi_ij = phi_ij[sort_idx, :]

    #     v_phi_theta= np.sum(phi_ij[:self.M,:]*v_ij[:self.M,None]*self.theta[:,None], axis=0)

    #     return np.exp(v_phi_theta)
    
    # def eval(self, x,y):
    #     """
    #     Evaluate the sum for a given x, summing over all terms defined by theta and roots.
    #     x: 2D array of points (n, dim), where n is the number of points and dim is the dimension.
    #     Returns an array of evaluations for each point.
    #     """
    #     # Initialize the result array (same size as the number of points)
    #     result = np.zeros(x.shape[0])
        
    #     # Compute the sum over all terms
    #     for i, (kx, ky) in enumerate(self.eigen_pairs):
    #         v_ij = self.an[kx] *self.an[ky]
    #         phi_i = self.A[kx] * (np.sin(self.roots[kx] * x) + (self.roots[kx] / 4) * np.cos(self.roots[kx] * x))
    #         phi_j = self.A[ky] * (np.sin(self.roots[ky] * y) + (self.roots[ky] / 4) * np.cos(self.roots[ky] * y))

    #         result += self.theta[i] * v_ij * phi_i*phi_j
    #     return np.exp(result)

class Elliptic2D_FEM:
    def __init__(self, theta, ncells=10):
        self.theta = theta

        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, nx =ncells, ny = ncells,cell_type=mesh.CellType.triangle )
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        self.bc = self.set_boundary_conditions()
        self.k = self.interpolate_k()
        self.f = self.interpolate_f()
        self.uh = None
    
    
    def interpolate_k(self):
        """Interpolate the k function based on the provided equation."""
        k = fem.Function(self.V)
        #k_an = Parametric_K(self.theta, l=self.l, mean=self.mean)
        k_an = Parametric_K(self.theta)
        k.interpolate(lambda x: k_an.eval(x[0], x[1]))
        return k
    
    def interpolate_f(self):
        """Interpolate the f function."""
        f = fem.Function(self.V)
        f.interpolate(lambda x: 4 * (x[0]*x[1]))
        return f

    def set_boundary_conditions(self):
        facets = mesh.locate_entities_boundary(self.domain, self.domain.topology.dim - 1,
                marker=lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0) | np.isclose(x[1], 1), )

        dofs = fem.locate_dofs_topological(V=self.V, entity_dim=1, entities=facets)
        # DirichletBC on all boundary
        bc = fem.dirichletbc(value=fem.Constant(self.domain, 0.0), dofs=dofs, V=self.V)
        return [bc]


    def solve(self):
        """Define and solve the linear variational problem."""
        # Interpolate k and f functions every time we solve
        self.k = self.interpolate_k()  # Ensure k is updated based on current theta
        self.f = self.interpolate_f()  # f can remain static, but can be updated if needed

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        # Define the bilinear form (a) and the linear form (L)
        a = self.k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = self.f *v * ufl.dx

        # Solve the linear problem
        problem = LinearProblem(a, L, bcs=self.bc)
        self.uh = problem.solve()
        return self.uh
    
    def solution_array(self):
        cells, types, x = plot.vtk_mesh(self.V)
        return (x, self.uh.x.array)
    
    def evaluate_at_points(self, points):
        """
        Evaluate the current and initial vorticity fields at specified points.
        """
        # Check if points array has correct shape
        if points.shape[1] != 3:
            points = np.hstack((points, np.zeros((points.shape[0], 1))))

        # Build the bounding box tree for the mesh
        bb_tree = geometry.bb_tree(self.domain, self.domain.topology.dim)

        # Find cells whose bounding boxes collide with the points
        cell_candidates = geometry.compute_collisions_points(bb_tree, points)

        # Compute colliding cells
        colliding_cells = geometry.compute_colliding_cells(self.domain, cell_candidates, points)

        # Store points and their corresponding cells
        cells = []
        points_on_proc = []

        for i, point in enumerate(points):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])  # Select the first colliding cell

        points_on_proc = np.array(points_on_proc, dtype=np.float64)

        if len(points_on_proc) > 0:
            # Evaluate `w` (current vorticity) and `w0` (initial vorticity) at the points
            u_values = self.uh.eval(points_on_proc, cells)
            return u_values
        else:
            raise ValueError("No points found on this process for evaluation.")