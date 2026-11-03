"""Compare deployed backend camera with the bundled camera on a calibration."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from egosmplx_pipeline.camera import FishEyeCameraCalibrated
from egosmplx_pipeline.projection import stable_world2camera
from egosmplx_pipeline.io import sha


def compare(backend_file, calibration):
    spec = importlib.util.spec_from_file_location('downsight_backend_fisheye', str(backend_file))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    a = FishEyeCameraCalibrated(calibration); b = module.FishEyeCameraCalibrated(calibration)
    w, h = a.img_size
    grid_x, grid_y = np.meshgrid(np.linspace(0, w-1, 17), np.linspace(0, h-1, 11))
    pixels = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    depth = np.linspace(0.3, 2.0, len(pixels))
    points = a.camera2world(pixels, depth)
    # Projection implementations reject an axis point, so use non-axis samples.
    if np.any(np.linalg.norm(points[:, :2],axis=1) == 0):
        points = points[np.linalg.norm(points[:, :2],axis=1) != 0]
    pt = torch.tensor(points, dtype=torch.float32)
    pixel_t = torch.tensor(pixels,dtype=torch.float32)
    dep_t = torch.tensor(depth,dtype=torch.float32)
    def maxdiff(x,y): return float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
    result = {
        'calibration_sha256': sha(calibration), 'backend_camera_sha256': sha(backend_file),
        'bundled_camera_sha256': sha(Path(__file__).resolve().parents[1]/'egosmplx_pipeline/camera.py'),
        'pixel_samples':len(pixels), 'point_samples':len(points),
        'backprojection_numpy_max_m':maxdiff(a.camera2world(pixels,depth),b.camera2world(pixels,depth)),
        'ray_numpy_max':maxdiff(a.camera2world_ray(pixels),b.camera2world_ray(pixels)),
        'backprojection_torch_max_m':maxdiff(a.camera2world_pytorch(pixel_t,dep_t).detach().numpy(),b.camera2world_pytorch(pixel_t,dep_t).detach().numpy()),
        'projection_numpy_max_px':maxdiff(a.world2camera(points),b.world2camera(points)),
        'projection_torch_class_max_px':maxdiff(a.world2camera_pytorch(pt).detach().numpy(),b.world2camera_pytorch(pt).detach().numpy()),
        'projection_torch_runtime_patch_vs_backend_max_px':maxdiff(stable_world2camera(a,pt).detach().numpy(),b.world2camera_pytorch(pt).detach().numpy()),
        'projection_backend_torch_vs_numpy_max_px':maxdiff(b.world2camera_pytorch(pt).detach().numpy(),b.world2camera(pt.numpy())),
        'projection_patch_vs_numpy_max_px':maxdiff(stable_world2camera(a,pt).detach().numpy(),b.world2camera(pt.numpy())),
        'scope':'CPU finite samples; no assertion of equality outside sampled points or GPU behavior'}
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--backend-camera',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True); a=p.parse_args()
    print(json.dumps(compare(a.backend_camera,a.calibration),indent=2))

if __name__=='__main__': main()
