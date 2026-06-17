import h5py
import os

# 查找h5文件
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.h5'):
            filepath = os.path.join(root, f)
            print(f"\n=== {filepath} ===")
            try:
                with h5py.File(filepath, 'r') as h5f:
                    print(f"Keys: {list(h5f.keys())}")
                    for key in h5f.keys():
                        data = h5f[key]
                        if hasattr(data, 'shape'):
                            print(f"  {key}: {data.shape}")
                        else:
                            print(f"  {key}: (group)")
            except Exception as e:
                print(f"Error: {e}")