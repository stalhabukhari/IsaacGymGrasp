import argparse
from isaacgymgrasp.grasp_quality_evaluation.grasps_sucess import GraspSuccessEvaluator

import numpy as np
import torch

import numpy as np
import torch
from pytorch3d.transforms.so3 import so3_rotation_angle
from scipy.optimize import linear_sum_assignment

try:
    from scripts.utils import read_hdf5_data, write_hdf5_data
except ImportError:
    from utils import read_hdf5_data, write_hdf5_data


def empirical_emd(H_data, H_sample=None, n_grasps=500, episodes=1000) -> tuple:
    """
    Earth Mover's Distance (EMD) between two empirical distributions
    H_data corresponds to good grasps for an object
    H_sample corresponds to generated grasps
    - If None, then sample n_grasps from H_data
    """
    # Set Samples
    if H_sample is None:
        # Data-Data EMD
        idx = np.random.randint(0, H_data.shape[0], n_grasps)
        H_sample = torch.Tensor(H_data[idx, ...]).float()
    else:
        H_sample = torch.Tensor(H_sample).float()
    p_sample = H_sample[:, :3, -1]
    R_sample = H_sample[:, :3, :3]

    divergence = np.zeros(0, dtype=np.float32)
    for k in range(episodes):
        ## Sample Candidates ##
        idx = np.random.randint(0, H_data.shape[0], n_grasps)
        H_eval = torch.Tensor(H_data[idx, ...]).to(H_sample)

        p_eval = H_eval[:, :3, -1]
        R_eval = H_eval[:, :3, :3]

        # translation distance
        xyz_dist = (p_eval[:, None, ...] - p_sample[None, ...]).pow(2).sum(-1).pow(0.5)

        # rotation distance
        R12 = torch.einsum("bmn,knd->bkmd", R_eval.transpose(-1, -2), R_sample)
        R12_ = R12.reshape(-1, 3, 3)
        R_dist_ = 1.0 - so3_rotation_angle(R12_, cos_angle=True)
        R_dist = R_dist_.reshape(R12.shape[0], R12.shape[1])

        # total
        distance = xyz_dist + R_dist

        distance = distance.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(distance)
        min_distance = distance[row_ind, col_ind].mean()
        divergence = np.concatenate((divergence, np.array([min_distance])), axis=0)

    mean = np.mean(divergence)
    std = np.std(divergence)
    return mean, std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_id", type=int, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    
    headless = False

    from pathlib import Path

    grasp_pred_dir = (
        Path("/media/talha/E0B28F75B28F4F4A/workspace/repositories/GIGAv2")
        / "vis/2025-01-05_16-05-27_Gdiff-exp2.3-DsdfV2In-LatentIn-NoVnnIn-noPE-Camera/ae_ckpt_10000_0.13448857069015502.pth-val"
    )

    obj_idx = args.obj_id
    h5_files = list(grasp_pred_dir.glob(f"{obj_idx}-*.h5"))
    assert len(h5_files) == 1
    h5_file = h5_files[0]

    # H = torch.eye(4).unsqueeze(0).repeat(1000, 1, 1)
    grasp_dict = read_hdf5_data(h5_file)
    H_pred = grasp_dict["grasp_H_pred"]
    H_gt = grasp_dict["grasp_H"]

    # compute success in simulation
    n_envs = H_pred.shape[0]
    evaluator = GraspSuccessEvaluator(
        data_dir="/media/talha/WD_BLACK/from-ideas-pc/GDIFF-data/",
        # n_envs=1,
        n_envs=n_envs,
        obj_class="Camera",
        idxs=[obj_idx],
        device="cuda:0",
        viewer=not args.headless,
        enable_rel_trafo=False,
    )

    H_pred_ = torch.from_numpy(H_pred)
    H_gt_ = torch.from_numpy(H_gt)

    #successes = evaluator.eval_set_of_grasps(H_pred_)
    successes = evaluator.eval_set_of_grasps(H_gt_)
    emd_self = empirical_emd(H_gt)
    emd_vs = empirical_emd(H_gt, H_pred)

    print(f"Success Rate: {successes} / {H_pred.shape[0]}")
    print(f"EMD Self: {emd_self[0]} +- {emd_self[1]}")
    print(f"EMD vs: {emd_vs[0]} +- {emd_vs[1]}")

    metrics_dict = {
        "obj_idx": obj_idx,
        "success_rate": successes / H_pred.shape[0],
        "emd_self": emd_self,
        "emd_vs": emd_vs,
    }

    # backing up as I go
    write_hdf5_data(f"metrics/{obj_idx}-metrics.h5", metrics_dict)
