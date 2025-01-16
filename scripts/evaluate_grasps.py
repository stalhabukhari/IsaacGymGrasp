from isaacgymgrasp.grasp_quality_evaluation.grasps_sucess import GraspSuccessEvaluator

# from isaacgymgrasp.utils.se3dif_utils import

import numpy as np
import torch

if __name__ == "__main__":
    headless = True
    evaluator = GraspSuccessEvaluator(
        data_dir="/media/talha/WD_BLACK/from-ideas-pc/GDIFF-data/",
        obj_class="Camera",
        n_envs=100,
        device="cuda:0",
        viewer=not headless,
    )

    H = torch.eye(4).unsqueeze(0).repeat(1000, 1, 1)
    successes = evaluator.eval_set_of_grasps(H)

    print(f"Success Rate: {successes} / {H.shape[0]}")
