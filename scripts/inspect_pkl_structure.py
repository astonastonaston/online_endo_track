import argparse
import pickle
import numpy as np

def inspect_pkl_structure(pkl_file, verbose=False):
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    print(f"\n[Info] Loaded {len(data)} frames from: {pkl_file}")
    print(f"[Type] Top-level object type: {type(data)}")

    if isinstance(data, list) and len(data) > 0:
        print("\n[Structure of First Frame]")
        first = data[0]
        for k, v in first.items():
            if isinstance(v, np.ndarray):
                print(f"  {k:15}: np.ndarray, shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k:15}: {type(v)}")
        
        if verbose:
            print("\n[Sample Values]")
            for k in first:
                print(f"  {k}: {first[k]}")
    else:
        print("Unsupported structure. Expected a list of dicts (one per frame).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect structure of a .pkl file")
    parser.add_argument("--file", type=str, help="Path to the .pkl file")
    parser.add_argument("--verbose", action="store_true", help="Print actual values for first frame")
    args = parser.parse_args()

    inspect_pkl_structure(args.file, args.verbose)

