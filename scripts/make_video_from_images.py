import imageio
import os
import argparse

def create_video_from_images(image_dir, output_video, start_idx=8521, end_idx=9715, step=6, fps=30):
    # Collect valid image filenames
    image_paths = []
    for i in range(start_idx, end_idx + 1, step):
        filename = f"{i:06d}l.png"
        full_path = os.path.join(image_dir, filename)
        if os.path.exists(full_path):
            image_paths.append(full_path)
        else:
            print(f"Warning: File not found: {filename}")

    if not image_paths:
        print("No valid images found.")
        return

    # Load images
    images = [imageio.imread(p) for p in image_paths]

    # Write to video
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    imageio.mimsave(output_video, images, fps=fps)
    print(f"Saved video to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from image sequence.")
    parser.add_argument("--image_dir", required=True, help="Directory containing image frames")
    parser.add_argument("--output", required=True, help="Output video path (e.g., output/video.mp4)")
    parser.add_argument("--start", type=int, default=8521, help="Start index (default: 8521)")
    parser.add_argument("--end", type=int, default=9715, help="End index (default: 9715)")
    parser.add_argument("--step", type=int, default=6, help="Step size (default: 6)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")

    args = parser.parse_args()

    create_video_from_images(
        image_dir=args.image_dir,
        output_video=args.output,
        start_idx=args.start,
        end_idx=args.end,
        step=args.step,
        fps=args.fps,
    )

# python make_video_from_images.py \
#     --image_dir ../../datasets/StereoMIS/P1_1/video_frames \
#     --output ../../datasets/StereoMISgt/P1_1/gt.mp4 \
#     --start 14341 --end 15535 --step 6 --fps 30

# python make_video_from_images.py \
#     --image_dir ../../datasets/StereoMIS/P2_1/video_frames \
#     --output ../../datasets/StereoMISgt/P2_1/gt.mp4 \
#     --start 5971 --end 7165 --step 6 --fps 30

# python make_video_from_images.py \
#     --image_dir ../../datasets/StereoMIS/P2_0/video_frames \
#     --output ../../datasets/StereoMISgt/P2_0/gt.mp4 \
#     --start 10009 --end 11203 --step 6 --fps 30

# python make_video_from_images.py \
#     --image_dir ../../datasets/StereoMIS/P2_2/video_frames \
#     --output ../../datasets/StereoMISgt/P2_2/gt.mp4 \
#     --start 1 --end 1195 --step 6 --fps 30

# python make_video_from_images.py \
#     --image_dir ../../datasets/StereoMIS/P2_6/video_frames \
#     --output ../../datasets/StereoMISgt/P2_6/gt.mp4 \
#     --start 10435 --end 11032 --step 6 --fps 30

# python make_video_from_images.py \
#     --image_dir ../../datasets/StereoMIS/P3_1/video_frames \
#     --output ../../datasets/StereoMISgt/P3_1/gt.mp4 \
#     --start 8521 --end 9715 --step 6 --fps 30

# python make_video_from_images.py \
#     --image_dir ../../datasets/StereoMIS/P3_2/video_frames \
#     --output ../../datasets/StereoMISgt/P3_2/gt.mp4 \
#     --start 10801 --end 11995 --step 6 --fps 30
