import os
import glob
import argparse
import numpy as np
import pickle
from tqdm import tqdm

def save_all_gaussians_to_pkl(npz_dir, output_file):
    files = sorted(glob.glob(os.path.join(npz_dir, "frame_*.npz")))
    if not files:
        raise FileNotFoundError("No frame_*.npz files found in the directory.")

    all_data = []

    for f in tqdm(files, desc="Loading frames"):
        data = np.load(f)
        frame_id = int(os.path.basename(f).split("_")[1].split(".")[0])
        frame_data = {
            "frame_id": frame_id,
            "xyz": data["deformed_xyz"],
            "rgb": data["rgb"] if "rgb" in data else np.ones_like(data["deformed_xyz"]) * 0.8,
            "opacity": data["opacity"] if "opacity" in data else np.ones((data["deformed_xyz"].shape[0], 1))
        }
        all_data.append(frame_data)

    with open(output_file, "wb") as f:
        pickle.dump(all_data, f)

    print(f"[Save] Stored {len(all_data)} frames to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stack all Gaussian data into a single .pkl file")
    parser.add_argument("--npz_dir", type=str, required=True, help="Directory containing frame_*.npz files")
    parser.add_argument("--output", type=str, default="all_gaussians.pkl", help="Output pkl file")
    args = parser.parse_args()

    save_all_gaussians_to_pkl(args.npz_dir, args.output)
