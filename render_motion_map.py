import os
import math
import torch
import numpy as np
import argparse
import cv2
import pickle
import time

from src.utils.datasets import StereoMIS  # your provided StereoMIS dataset file
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class SimpleCamera:
    """ Minimal camera holder compatible with the rasterizer """
    def __init__(self, H, W, focal, pose):
        self.image_height = H
        self.image_width = W
        self.FoVx = focal2fov(focal, W)
        self.FoVy = focal2fov(focal, H)

        # pose is 4x4 world-to-camera transform
        self.world_view_transform = torch.linalg.inv(pose).cuda()
        self.full_proj_transform = self.world_view_transform.clone()  # projection can be appended later
        self.camera_center = pose[:3, 3].cuda()


def render_flow_map(camera, means, covs, opacities, flow_vectors, bg_color):
    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)

    # Flow magnitude -> grayscale RGB
    flow_magnitude = torch.norm(flow_vectors, dim=1, keepdim=True)
    flow_color = flow_magnitude.expand(-1, 3).clamp(0.0, 1.0)

    world_view_transform = camera.world_view_transform
    world_view_transform = world_view_transform.cuda()
    full_proj_transform = camera.full_proj_transform
    full_proj_transform = full_proj_transform.cuda()
    camera_center = camera.camera_center
    camera_center = camera_center.cuda()
    print("world ", world_view_transform.shape, world_view_transform.device)
    print("full ", full_proj_transform.shape, full_proj_transform.device)
    print("cam cen ", camera_center.shape, camera_center.device)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False  # no pipe, just hardcoded
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    print(means.shape, means.device, flow_color.shape, flow_color.device, opacities.shape, opacities.device, covs.shape, covs.device)
    rendered, _, _, _, _ = rasterizer(
        means3D=means,
        means2D=torch.zeros_like(means[:, :2], device="cuda"),
        shs=None,
        colors_precomp=flow_color,
        opacities=opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=covs
    )
    print("rednered", rendered, rendered.shape)


    return rendered


def save_motion_map_for_frame(dataset, motion_pkl, output_dir, frame_id, H, W, focal):
    # Step 1: Load pose from StereoMIS
    _, _, _, pose, _, _ = dataset[frame_id]
    pose = pose.cuda()

    # Step 2: Make camera object
    camera = SimpleCamera(H, W, focal, pose)

    # Step 3: Load trajectory gaussians and flows from pkl
    with open(motion_pkl, 'rb') as f:
        motion_data = pickle.load(f)

    frame_data = motion_data[frame_id]

    means = torch.tensor(frame_data['traj_means'], dtype=torch.float32, device="cuda")
    covs = torch.tensor(frame_data['traj_covariances'], dtype=torch.float32, device="cuda")
    covs = strip_symmetric(covs)
    flow_vectors = torch.tensor(frame_data['motion_flows'], dtype=torch.float32, device="cuda")

    # Opacities placeholder
    opacities = torch.ones((means.shape[0], 1), dtype=torch.float32, device="cuda") * 0.6

    # Step 4: Render motion map
    rendered = render_flow_map(
        camera=camera,
        means=means,
        covs=covs,
        opacities=opacities,
        flow_vectors=flow_vectors,
        bg_color=torch.tensor([1.0, 1.0, 1.0], device="cuda")
    )
    # time.sleep(5)
    # print(rendered)
    rendered = rendered.cpu().numpy()

    # Step 5: Save
    rendered_clamped = (rendered.clamp(0, 1) * 255).byte()
    rendered_np = rendered_clamped.permute(1, 2, 0).cpu().numpy()
    rendered_np = np.ascontiguousarray(rendered_np, dtype=np.uint8)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"motion_frame_{frame_id:04d}.png")
    cv2.imwrite(out_path, cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR))
    print(f"[Done] Saved motion map to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render motion flow map from StereoMIS trajectory Gaussians")
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--motion_pkl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frame_id", type=int, required=True)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--focal", type=float, default=500.0)
    args = parser.parse_args()

    # Dummy config to init StereoMIS
    cfg = {
        "dataset": "StereoMIS",
        "data": {
            "input_folder": args.input_folder,
            "start": 0,
            "stop": None,
            "step": 1
        }
    }
    class Args:
        pass
    args_ns = Args()
    args_ns.input_folder = args.input_folder

    dataset = StereoMIS(cfg, args_ns, scale=1.0)

    save_motion_map_for_frame(
        dataset=dataset,
        motion_pkl=args.motion_pkl,
        output_dir=args.output_dir,
        frame_id=args.frame_id,
        H=args.H,
        W=args.W,
        focal=args.focal
    )


if __name__ == "__main__":
    main()


# python render_motion_map.py \
#     --input_folder /home/nan/Desktop/datasets/StereoMIS/P3_1 \
#     --motion_pkl /home/nan/Desktop/gaussians_StereoMIS/exports/all_traj_gaussians_and_flows.pkl \
#     --output_dir ../output/motion_vis \
#     --frame_id 10
    # --H 1080 \
    # --W 1920 \
    # --focal 1200



