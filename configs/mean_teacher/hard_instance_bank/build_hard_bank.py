import pickle
from pathlib import Path
from mmdet3d.datasets.transforms.hard_instance_mining import build_hard_instance_bank

# ============================================================
# Build Hard Instance Bank (Run Once Before Training)
# ============================================================

source_db_path = '/DATA/nuScenes/nuscenes_dbinfos_train.pkl'
target_db_path = '/DATA/kitti_mmdet3d/kitti_dbinfos_train.pkl'
quantile_threshold = 20
source_dataset_name = 'nuscenes'
target_dataset_name = 'kitti'

# Base path for source GT database files
source_gt_path_base = Path('/DATA/nuScenes/temp')

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
    target_db_path=target_db_path,
    quantile_threshold=quantile_threshold,
    classes=list(source_class_mapping.keys()),
    source_class_mapping=source_class_mapping
)

# Convert relative paths to absolute (fix double path issue)
print("Converting relative paths to absolute...")
count = 0
for cls_name in hard_bank.hard_instances:
    for sample in hard_bank.hard_instances[cls_name]:
        if 'path' in sample:
            sample_path = Path(sample['path'])
            
            # If path is relative, make it absolute
            if not sample_path.is_absolute():
                # The path is like "nuscenes_gt_database/2_car_3.bin"
                # Just prepend the base path
                sample['path'] = str(source_gt_path_base / sample_path)
                count += 1
            else:
                # If already absolute but has double directory, fix it
                path_str = str(sample_path)
                if 'nuscenes_gt_database/nuscenes_gt_database' in path_str:
                    # Remove one occurrence of the duplicate
                    fixed_path = path_str.replace('nuscenes_gt_database/nuscenes_gt_database', 'nuscenes_gt_database')
                    sample['path'] = fixed_path
                    count += 1

print(f"✅ Converted {count} paths.")

# Save the bank
save_path = Path(f"./configs/mean_teacher/hard_instance_bank/hard_instance_bank_{source_dataset_name}_quantile_{target_dataset_name}_{quantile_threshold}.pkl")
save_path.parent.mkdir(parents=True, exist_ok=True)

with open(save_path, 'wb') as f:
    pickle.dump(hard_bank, f)
    print(f"✅ Hard instance bank saved to {save_path}")

# Verify
print("\nVerification:")
for cls_name, samples in hard_bank.hard_instances.items():
    if samples:
        print(f"  {cls_name}: {len(samples)} hard instances")
        sample_path = samples[0]['path']
        exists = Path(sample_path).exists()
        status = "✅" if exists else "❌"
        print(f"    {status} Sample path: {sample_path}")