# pieces of code ported from se3dif (https://github.com/robotgradient/grasp_diffusion)

import os
import glob
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import trimesh
import h5py
from theseus.geometry import SO3


# ------------------ math ------------------
class SO3_R3:
    def __init__(self, R=None, t=None):
        self.R = SO3()
        if R is not None:
            self.R.update(R)
        self.w = self.R.log_map()
        if t is not None:
            self.t = t

    def log_map(self):
        return torch.cat((self.t, self.w), -1)

    def exp_map(self, x):
        self.t = x[..., :3]
        self.w = x[..., 3:]
        self.R = SO3().exp_map(self.w)
        return self

    def to_matrix(self):
        H = torch.eye(4).unsqueeze(0).repeat(self.t.shape[0], 1, 1).to(self.t)
        H[:, :3, :3] = self.R.to_matrix()
        H[:, :3, -1] = self.t
        return H

    # The quaternion takes the [w x y z] convention
    def to_quaternion(self):
        return self.R.to_quaternion()

    def sample(self, batch=1):
        R = SO3().rand(batch)
        t = torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)
        H[:, :3, :3] = R.to_matrix()
        H[:, :3, -1] = t
        return H


# ------------------ dataset ------------------
class AcronymGrasps:
    def __init__(self, filename, data_dir=None):
        self.data_dir = data_dir

        scale = None
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            self.mesh_fname = data["object"].decode("utf-8")
            self.mesh_type = self.mesh_fname.split("/")[1]
            self.mesh_id = self.mesh_fname.split("/")[-1].split(".")[0]
            self.mesh_scale = data["object_scale"] if scale is None else scale
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            self.mesh_fname = data["object/file"][()].decode("utf-8")
            self.mesh_type = self.mesh_fname.split("/")[1]
            self.mesh_id = self.mesh_fname.split("/")[-1].split(".")[0]
            self.mesh_scale = data["object/scale"][()] if scale is None else scale
        else:
            raise RuntimeError("Unknown file ending:", filename)

        self.grasps, self.success = self.load_grasps(filename)
        good_idxs = np.argwhere(self.success == 1)[:, 0]
        bad_idxs = np.argwhere(self.success == 0)[:, 0]
        self.good_grasps = self.grasps[good_idxs, ...]
        self.bad_grasps = self.grasps[bad_idxs, ...]

    def load_grasps(self, filename):
        """Load transformations and qualities of grasps from a JSON file from the dataset.

        Args:
            filename (str): HDF5 or JSON file name.

        Returns:
            np.ndarray: Homogenous matrices describing the grasp poses. 2000 x 4 x 4.
            np.ndarray: List of binary values indicating grasp success in simulation.
        """
        if filename.endswith(".json"):
            data = json.load(open(filename, "r"))
            T = np.array(data["transforms"])
            success = np.array(data["quality_flex_object_in_gripper"])
        elif filename.endswith(".h5"):
            data = h5py.File(filename, "r")
            T = np.array(data["grasps/transforms"])
            success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        else:
            raise RuntimeError("Unknown file ending:", filename)
        return T, success

    def load_mesh(self):
        mesh_path_file = os.path.join(self.data_dir, self.mesh_fname)

        mesh = trimesh.load(mesh_path_file, file_type="obj", force="mesh")

        mesh.apply_scale(self.mesh_scale)
        if type(mesh) == trimesh.scene.scene.Scene:
            mesh = trimesh.util.concatenate(mesh.dump())
        return mesh

    def sampled_pc_filename_from_mesh_fname(self):
        mesh_filepath = Path(os.path.join(self.data_dir, self.mesh_fname))
        obj_instance = mesh_filepath.stem
        obj_class = mesh_filepath.parent.stem

        mesh_sampled_pc_dir = Path(self.data_dir) / "mesh_sampled_pc"
        pc_filepath = mesh_sampled_pc_dir / obj_class / f"{obj_instance}.npy"
        return pc_filepath

    def write_mesh_sampled_pc(self, n_points=5000):
        mesh = self.load_mesh()
        pc = mesh.sample(n_points)

        npy_filepath = self.sampled_pc_filename_from_mesh_fname()
        if not npy_filepath.parent.exists():
            if not npy_filepath.parent.parent.exists():
                npy_filepath.parent.parent.mkdir(parents=False, exist_ok=True)
            npy_filepath.parent.mkdir(parents=False, exist_ok=True)
        print(f"Saving sampled point cloud {pc.shape} to: {npy_filepath}")
        np.save(npy_filepath, pc)

    def sdf_filename_from_mesh_fname(self):
        mesh_fname = self.mesh_fname
        mesh_type = mesh_fname.split("/")[1]
        mesh_name = mesh_fname.split("/")[-1]
        filename = mesh_name.split(".obj")[0]
        sdf_filepath = Path(self.data_dir) / "sdf" / mesh_type / (filename + ".json")
        return sdf_filepath

    def write_sdf_npy_from_json_file(self):
        mesh_scale = self.mesh_scale
        sdf_filepath = self.sdf_filename_from_mesh_fname()
        with open(sdf_filepath, "rb") as f:
            sdf_dict = pickle.load(f)

        loc = sdf_dict["loc"]
        scale = sdf_dict["scale"]
        xyz = (sdf_dict["xyz"] + loc) * scale * mesh_scale
        sdf = sdf_dict["sdf"] * scale * mesh_scale

        xyz_sdf = np.hstack([xyz, sdf.reshape(-1, 1)]).astype(np.float32)
        npy_filepath = sdf_filepath.parent / f"{sdf_filepath.stem}.npy"
        print(f"Saving SDF coords,values {xyz_sdf.shape} to: {npy_filepath}")
        np.save(npy_filepath, xyz_sdf)


def get_grasps_acr(data_dir, class_type):
    grasps_dir = os.path.join(data_dir, "grasps")
    print(f"> Searching from grasps under: {grasps_dir}")
    grasp_objs = []
    for class_type_i in class_type:
        cls_grasps_files = sorted(glob.glob(grasps_dir + "/" + class_type_i + "/*.h5"))

        for grasp_file in cls_grasps_files:
            g_obj = AcronymGrasps(grasp_file, data_dir=data_dir)
            if g_obj.good_grasps.shape[0] == 0:
                print(f"> {g_obj.mesh_fname} has no good grasps")
                continue
            grasp_objs.append(g_obj)
    return grasp_objs
