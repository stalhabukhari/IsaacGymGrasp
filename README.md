# Isaac Gym Environment for Simulating Robot Grasp

## Setup

- Create a python 3.8 conda environment.
- Install PyTorch (and a few other things):
```sh
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install -c conda-forge suitesparse==5.10.1 scikit-sparse==0.4.8
pip install cmake lit
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
- Install IsaacGymEnvs: https://github.com/stalhabukhari/IsaacGymEnvs.git
    - For `libpython3.8.so.1.0` error: Copy the file from conda directory to /usr/lib/ or /usr/lib64/ ([reference: stackoverflow](https://stackoverflow.com/a/74725104))
- Then do:
```sh
pip install -r requirements.txt
```
- And then, setup the repository as:
```sh
pip install -e . --use-pep517
```
- For recording videos, install libx264:
```sh
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
```

## Dataset

- This repository assumes the Acronym dataset structure.

TODO: Add structure that assumes dataset in the format of an `.h5` file, comprising the following structure:

```
{
    'objcat/objid': np.array,
    ...
}
```

TODO: add logic for `gripper fully closed -> failure`

## Execute

Evaluate object grasps via:

```sh
python scripts/evaluate_grasps.py --obj_id 7;
```

## References

Repository is adapted from: https://github.com/robotgradient/grasp_diffusion
