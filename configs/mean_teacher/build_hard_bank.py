
from mmdet3d.datasets.transforms.hard_instance_mining import build_hard_instance_bank

# ============================================================
# 1. Build Hard Instance Bank (Run Once Before Training)
# ============================================================

# Paths to your database files
### EDIT HERE BEFORE RUN ###
source_db_path = '/DATA/nuscenes/nuscenes_dbinfos_train.pkl'
target_db_path = '/DATA/kitti_mmdet3d/kitti_dbinfos_train.pkl'
quantile_threshold = 50  # γ = 50 (median split)
target_dataset_name = 'kitti'

# Build the hard instance bank
hard_bank = build_hard_instance_bank(
    source_db_path=source_db_path,
    target_db_path=target_db_path,  # For adaptive thresholding
    quantile_threshold=quantile_threshold,  # γ = 50 (median split)
    classes=['Car', 'Pedestrian', 'Cyclist']
)

# Save the bank for reuse (optional)
import pickle
with open(f"hard_instance_bank_{target_dataset_name}_quantile_{quantile_threshold}.pkl", 'wb') as f:
    pickle.dump(hard_bank, f)