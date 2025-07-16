import imageio
import os

# Parameters
start_idx = 8521
end_idx = 9715
step = 6
image_dir = '../datasets/StereoMIS/P3_1/video_frames'  # directory containing the images
output_video = 'output/output.mp4'
fps = 30  # frames per second

# Collect valid image filenames
image_paths = []
for i in range(start_idx, end_idx + 1, step):
    filename = f"{i:06d}l.png"
    full_path = os.path.join(image_dir, filename)
    if os.path.exists(full_path):
        image_paths.append(full_path)
    else:
        print(f"Warning: File not found: {filename}")

# Load images
images = [imageio.imread(p) for p in image_paths]

# Write to video
imageio.mimsave(output_video, images, fps=fps)
print(f"Saved video to {output_video}")
