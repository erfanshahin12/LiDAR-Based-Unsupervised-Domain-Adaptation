import torch
from pathlib import Path

# --------------------------------------------------
# Config
# --------------------------------------------------
ckpt_path = "/home/erfans00/mmdetection3d/checkpoints/pvrcnn_cmt.pth"   # <-- CHANGE THIS
print_shapes = True            # set False if too verbose
max_keys_per_section = 20      # safety limit for printing

# --------------------------------------------------
# Load checkpoint (CPU-safe)
# --------------------------------------------------
ckpt_path = Path(ckpt_path)
assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

checkpoint = torch.load(ckpt_path, map_location="cpu")

print("\n==============================")
print("Checkpoint loaded successfully")
print("==============================\n")

# --------------------------------------------------
# Top-level keys
# --------------------------------------------------
print("Top-level keys in checkpoint:")
for k in checkpoint.keys():
    print(f"  - {k}")

# --------------------------------------------------
# Inspect state_dict
# --------------------------------------------------
if "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]

    print("\n==============================")
    print("STATE_DICT")
    print("==============================")
    print(f"Number of parameters: {len(state_dict)}\n")

    for i, (name, tensor) in enumerate(state_dict.items()):
        if i >= max_keys_per_section:
            print("  ... (truncated)")
            break

        if torch.is_tensor(tensor):
            if print_shapes:
                print(f"  {name:<80} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
            else:
                print(f"  {name}")
        else:
            print(f"  {name}: {type(tensor)}")

else:
    print("\n❌ No 'state_dict' found in checkpoint")

# --------------------------------------------------
# Inspect optimizer state
# --------------------------------------------------
if "optimizer_state" in checkpoint:
    print("\n==============================")
    print("OPTIMIZER STATE")
    print("==============================")

    opt = checkpoint["optimizer_state"]
    print(f"Optimizer keys: {list(opt.keys())}")

    if "param_groups" in opt:
        print(f"Number of param groups: {len(opt['param_groups'])}")
        for i, g in enumerate(opt["param_groups"][:max_keys_per_section]):
            print(f"  Param group {i}:")
            for k, v in g.items():
                if k != "params":
                    print(f"    {k}: {v}")

# --------------------------------------------------
# Inspect meta information (MMDet/MMDet3D)
# --------------------------------------------------
if "meta" in checkpoint:
    print("\n==============================")
    print("META INFORMATION")
    print("==============================")

    meta = checkpoint["meta"]
    for k, v in meta.items():
        print(f"  {k}: {v}")

# --------------------------------------------------
# Epoch / iteration
# --------------------------------------------------
for key in ["epoch", "iter", "iteration", "it"]:
    if key in checkpoint:
        print(f"\n{key}: {checkpoint[key]}")

# --------------------------------------------------
# Other fields
# --------------------------------------------------
known_keys = {"state_dict", "optimizer", "meta", "epoch", "iter", "iteration"}
extra_keys = set(checkpoint.keys()) - known_keys

if extra_keys:
    print("\n==============================")
    print("OTHER FIELDS")
    print("==============================")
    for k in extra_keys:
        print(f"  {k}: type={type(checkpoint[k])}")

print("\n==============================")
print("Inspection finished")
print("==============================")

with open("pvrcnn_cmt.txt", "w") as f:
    for k in state_dict:
        f.write(k + "\n")