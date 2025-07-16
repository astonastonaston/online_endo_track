import pickle
import numpy as np
import argparse

def print_structure(obj, indent=0, key_prefix="root"):
    print(obj)
    prefix = "  " * indent + f"{key_prefix}: "
    if isinstance(obj, dict):
        print(f"{prefix}dict with {len(obj)} keys")
        for k, v in obj.items():
            print_structure(v, indent + 1, key_prefix=str(k))
    elif isinstance(obj, list):
        print(f"{prefix}list with {len(obj)} elements")
        for i, v in enumerate(obj[:5]):  # Show only first 5 items
            print_structure(v, indent + 1, key_prefix=f"[{i}]")
        if len(obj) > 5:
            print("  " * (indent + 1) + "... (truncated)")
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}ndarray, shape: {obj.shape}, dtype: {obj.dtype}")
    elif hasattr(obj, 'shape') and hasattr(obj, 'dtype'):  # e.g. torch.Tensor
        print(f"{prefix}{type(obj).__name__}, shape: {obj.shape}, dtype: {obj.dtype}")
    else:
        val = str(obj)
        print(f"{prefix}{type(obj).__name__}: {val[:80]}{'...' if len(val) > 80 else ''}")

def main():
    parser = argparse.ArgumentParser(description="Inspect structure of tracked.pckl")
    parser.add_argument("path", type=str, help="Path to tracked.pckl file")
    args = parser.parse_args()

    print(f"\nLoading and inspecting: {args.path}")
    with open(args.path, "rb") as f:
        data = pickle.load(f)
    print(data.keys())
    print(len(data['pred_2d']))
    print(data['pred_2d'][0])
    # print_structure(data)

if __name__ == "__main__":
    main()

# python inspect_tracked_pckl.py output/stereoMIS/P3_1_explicit/tracked.pckl
