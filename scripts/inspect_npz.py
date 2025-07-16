import numpy as np
import argparse

def inspect_npz(file_path):
    try:
        data = np.load(file_path)
        print(f"\n📂 Inspecting NPZ file: {file_path}")
        print("-" * 50)
        for key in data.files:
            value = data[key]
            print(f"Key: '{key}'")
            print(f"  Type: {type(value)}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Shape: {value.shape}")
            print("-" * 50)
    except Exception as e:
        print(f"❌ Error loading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect structure of a .npz file")
    parser.add_argument("--file", type=str, help="Path to the .npz file")
    args = parser.parse_args()

    inspect_npz(args.file)
