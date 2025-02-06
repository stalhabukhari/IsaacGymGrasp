import argparse
from pathlib import Path

from isaacgymgrasp.grasp_quality_evaluation.grasps_sucess import GraspSuccessEvaluator

import numpy as np
import torch

try:
    from scripts.utils import write_yaml_file
except ImportError:
    from utils import write_yaml_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_cat", type=str, required=True)
    parser.add_argument("--obj_id", type=str, required=True)
    parser.add_argument("--grasp_file", type=Path, required=True)
    parser.add_argument("--n_envs", type=int, default=1, required=False)
    parser.add_argument("--n_grasps", type=int, default=None, required=False)
    parser.add_argument(
        "--objs_data_dir",
        type=Path,
        required=False,
        default=Path("/media/talha/WD_BLACK/from-ideas-pc/GDIFF-data/"),
        help="Path to object meshes",
    )
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    assert args.grasp_file.exists(), f"Grasp file {args.grasp_file} does not exist"
    assert ".npy" in args.grasp_file.name, "Grasp file must be in .npy format"
    assert (
        args.metrics_dir.exists()
    ), f"Metrics directory {args.metrics_dir} does not exist"

    # H = torch.eye(4).unsqueeze(0).repeat(1000, 1, 1)
    H_pred = np.load(args.grasp_file)
    if args.n_grasps is not None:
        H_pred = H_pred[: args.n_grasps]
    H_pred = torch.from_numpy(H_pred).float().to("cuda:0")

    # compute success in simulation
    assert H_pred.shape[0] % args.n_envs == 0, (
        "Number of environments must divide number of grasps"
        f"(found {H_pred.shape[0]} grasps for {args.n_envs} environments)"
    )

    evaluator = GraspSuccessEvaluator(
        data_dir=args.objs_data_dir,
        n_envs=args.n_envs,
        obj_class=args.obj_cat,
        obj_id=args.obj_id,
        rotations=None,
        device="cuda:0",
        viewer=not args.headless,
        enable_rel_trafo=False,
    )

    # evaluator.grasping_env.step()
    # for _ in range(1000):
    #     evaluator.grasping_env.step()
    # exit(0)

    successes = evaluator.eval_set_of_grasps(H_pred)

    print(f"****** Success Rate: {successes} / {H_pred.shape[0]} ******")
    # exit(0)

    metrics_dict = {
        "objcatid": f"{args.obj_cat}-{args.obj_id}",
        "total_grasps": H_pred.shape[0],
        "successes": successes,
        "success_rate": successes / H_pred.shape[0] * 100,
    }

    # write to disk
    metrics_fp = args.metrics_dir / f"{args.obj_cat}-{args.obj_id}-sr.yml"
    write_yaml_file(metrics_dict, metrics_fp)
