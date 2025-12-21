import pickle
from mmdet3d.datasets.transforms.hard_instance_mining import build_hard_instance_bank

# ============================================================
# 1. Build Hard Instance Bank (Run Once Before Training)
# ============================================================

# Paths to your database files
### EDIT HERE BEFORE RUN ###
source_db_path = '/DATA/nuscenes/nuscenes_dbinfos_train.pkl'
target_db_path = '/DATA/kitti_mmdet3d/kitti_dbinfos_train.pkl'
quantile_threshold = 50  # γ = 50 (median split)
source_dataset_name = 'nuscenes'
target_dataset_name = 'kitti'

# Class mapping: nuScenes → KITTI unified names
source_class_mapping = {
        'Car': 'car',
        'Pedestrian': 'pedestrian',
        'Cyclist': ['bicycle', 'motorcycle']
}

# Build the hard instance bank
print("Building hard instance bank...")
hard_bank = build_hard_instance_bank(
    source_db_path=source_db_path,
    target_db_path=target_db_path,  # For adaptive thresholding
    quantile_threshold=quantile_threshold,
    classes=list(source_class_mapping.keys()),
    source_class_mapping=source_class_mapping
)

save_path = f"./configs/mean_teacher/hard_instance_bank/hard_instance_bank_{source_dataset_name}_quantile_{target_dataset_name}_{quantile_threshold}.pkl"

# Save the bank for reuse (optional)
with open(save_path, 'wb') as f:
    pickle.dump(hard_bank, f)
    print(f"✅ Hard instance bank saved to {save_path}")

# Verify
print("\nVerification:")
for cls_name, samples in hard_bank.hard_instances.items():
    print(f"  {cls_name}: {len(samples)} hard instances")