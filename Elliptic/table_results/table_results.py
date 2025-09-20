import torch
import numpy as np
import json  # or import pickle if you prefer pickle
import sys
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)  # This allows importing from base, Elliptic, etc.
sys.path.append(os.path.join(project_root, "Elliptic"))  # Explicitly add Elliptic folder

from elliptic_files.utilities import *
from Base.utilities import *



# def run_experiment(nn_config, gp_config, kl, device="cpu"):
#     nn_res, gp_res = {}, {}

#     eval = samples_param(1000, kl)
#     obs_points, _ = generate_noisy_obs(obs=6, std=np.sqrt(1e-4), nparam=kl, vert=50)
#     obs_points = torch.tensor(obs_points)

#     if kl > 2:
#         model_specific = f"_hl2_nn100_s4000_batch{nn_config}_kl{kl}"
#     else:
#         model_specific = f"_hl2_nn20_s{nn_config}_batch100_kl{kl}"

#     path_nn_model = f"./Elliptic/models/elliptic{model_specific}.pth"
#     nn_surrogate_model = torch.load(path_nn_model, map_location=device)
#     nn_surrogate_model.eval()

#     data_training = pigp_training_data_generation(gp_config, 6, kl, device)
#     elliptic_gp = torch.load(f"./Elliptic/models/elliptic_pigp__spatial6_nsol{gp_config}_kl{kl}.pt", map_location=device)

#     # Timing
#     times_nn, times_gp = [], []
#     for evl in eval:
#         theta = torch.tensor(evl).reshape(1, -1)
#         data = torch.cat([obs_points, theta.repeat(obs_points.size(0), 1)], dim=1).float()

#         # NN eval
#         train_timer = Timer(use_gpu=True)
#         train_timer.start()
#         _ = nn_surrogate_model.u(data)
#         times_nn.append(train_timer.stop())

#         # GP eval
#         train_timer = Timer(use_gpu=True)
#         train_timer.start()
#         _ = elliptic_gp.prediction(theta, var=False)
#         times_gp.append(train_timer.stop())

#     nn_res["eval_time"] = (np.mean(times_nn), np.std(times_nn))
#     gp_res["eval_time"] = (np.mean(times_gp), np.std(times_gp))

#     # Individual results
#     nn_ind_res = {"post": [], "bound": []}
#     gp_ind_res = {"post": [], "bound": []}

#     for i in range(4):
#         chain_fem = np.load(f'./Elliptic/results/FEM_kl{kl}_var0.0001_{i}.npy')
#         fem_solver = FEMSolver(np.zeros(kl), M=kl, vert=50)

#         error_mcmc = error_norm_mean(nn_surrogate_model, fem_solver, obs_points, torch.tensor(chain_fem), device)
#         nn_ind_res["bound"].append(error_mcmc)

#         error_mcmc = error_norm_mean(elliptic_gp, fem_solver, obs_points, torch.tensor(chain_fem), device, gp=True)
#         gp_ind_res["bound"].append(error_mcmc)

#         nn_post_eval = np.load(f"./Elliptic/results/mcmc_da_nn{model_specific}_0.0001_{i}.npy")
#         gp_post_eval = np.load(f"./Elliptic/results/mcmc_da_pigp_mean_spatial6_nsol{gp_config}_kl{kl}_0.0001_{i}.npy")

#         nn_ind_res["post"].append(nn_post_eval.sum() / nn_post_eval.shape[-1])
#         gp_ind_res["post"].append(gp_post_eval.sum() / gp_post_eval.shape[-1])

#     # Aggregate
#     nn_res["post_eval"] = (np.mean(nn_ind_res["post"]), np.std(nn_ind_res["post"]))
#     gp_res["post_eval"] = (np.mean(gp_ind_res["post"]), np.std(gp_ind_res["post"]))
#     nn_res["m_error"] = (np.mean(nn_ind_res["bound"]), np.std(nn_ind_res["bound"]))
#     gp_res["m_error"] = (np.mean(gp_ind_res["bound"]), np.std(gp_ind_res["bound"]))

#     return kl, nn_res, gp_res


# if __name__ == "__main__":
#     NN_config = [500, 250, 250, 250]
#     GP_config = [25, 100, 100, 100]
#     KLs = [2, 3, 4, 5]

#     nn_results, gp_results = {}, {}

#     with ProcessPoolExecutor(max_workers=4) as executor:
#         futures = [executor.submit(run_experiment, nn_config, gp_config, kl, "cpu")
#                for nn_config, gp_config, kl in zip(NN_config, GP_config, KLs)]

#     for f in as_completed(futures):
#         kl, nn_res, gp_res = f.result()
#         nn_results[f"kl_{kl}"] = nn_res
#         gp_results[f"kl_{kl}"] = gp_res

#     # Save dictionaries once at the end
#     with open("nn_results.json", "w") as f:
#         json.dump(nn_results, f, indent=4)

#     with open("gp_results.json", "w") as f:
#         json.dump(gp_results, f, indent=4)

#     print("All results saved!")


def run_experiment(gp_config, kl, device="cpu"):
    gp_res = {}

    obs_points, _ = generate_noisy_obs(obs=6, std=np.sqrt(1e-4), nparam=kl, vert=50)
    obs_points = torch.tensor(obs_points)

    elliptic_gp = torch.load(f"./Elliptic/models/elliptic_pigp__spatial6_nsol{gp_config}_kl{kl}.pt", map_location=device)

    # Individual results
    gp_ind_res = {"post": [], "bound": []}

    for i in range(4):
        chain_fem = np.load(f'./Elliptic/results/FEM_kl{kl}_var0.0001_{i}.npy')
        fem_solver = FEMSolver(np.zeros(kl), M=kl, vert=50)

        error_mcmc = error_norm_mean(elliptic_gp, fem_solver, obs_points, torch.tensor(chain_fem), device, gp=True)
        gp_ind_res["bound"].append(error_mcmc)

        gp_post_eval = np.load(f"./Elliptic/results/mcmc_da_pigp_mean_spatial6_nsol{gp_config}_kl{kl}_0.0001_{i}.npy")

        gp_ind_res["post"].append(gp_post_eval.sum() / gp_post_eval.shape[-1])

    # Aggregate
    gp_res["post_eval"] = (np.mean(gp_ind_res["post"]), np.std(gp_ind_res["post"]))
    gp_res["m_error"] = (np.mean(gp_ind_res["bound"]), np.std(gp_ind_res["bound"]))

    return gp_config,kl, gp_res


if __name__ == "__main__":
    GP_config = [25, 50, 75, 90, 100]
    KLs = [3, 4, 5]

    gp_results = {}

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_experiment, gp_config, kl, "cpu")
               for gp_config in GP_config for kl in KLs]

    for f in as_completed(futures):
        gp_config, kl, gp_res = f.result()
        gp_results[f"kl_{kl}_{gp_config}"] = gp_res

    with open("gp_results.json", "w") as f:
        json.dump(gp_results, f, indent=4)

    print("All results saved!")