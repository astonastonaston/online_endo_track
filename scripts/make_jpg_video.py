import imageio
import os
import argparse

def generate_video_from_jpgs(image_dir, output_path, start_idx=0, end_idx=199, fps=30):
    image_paths = []
    for i in range(start_idx, end_idx + 1):
        filename = f"{i:05d}.jpg"
        full_path = os.path.join(image_dir, filename)
        if os.path.exists(full_path):
            image_paths.append(full_path)
        else:
            print(f"Warning: File not found: {filename}")

    if not image_paths:
        print("No valid images found.")
        return

    images = [imageio.imread(p) for p in image_paths]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, images, fps=fps)
    print(f"Saved video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from sequentially numbered JPG images.")
    parser.add_argument("--image_dir", required=True, help="Directory containing .jpg images")
    parser.add_argument("--output", required=True, help="Path to save the output video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    args = parser.parse_args()

    generate_video_from_jpgs(
        image_dir=args.image_dir,
        output_path=args.output,
        fps=args.fps,
    )

# python make_jpg_video.py --image_dir /home/nan/Desktop/online_endo_track/output/stereoMIS/P3_1_explicit_tissues/deltap --output /home/nan/Desktop/online_endo_track/output/stereoMIS/P3_1_explicit_tissues/deltap/video.mp4 --fps 30

