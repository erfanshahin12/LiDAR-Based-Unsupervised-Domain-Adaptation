import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_kitti(bin_path, output_path):
    # 1. Load Data
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    
    # DOWNSAMPLING NOTE: 
    # For very high-res images, you might want MORE points.
    # [::2] takes half the points. Use [::1] for all points (slower but denser).
    # points = points[::2]

    # Filter area (same as before)
    mask = (points[:, 0] > 0) & (points[:, 0] < 40) & \
           (np.abs(points[:, 1]) < 20) & \
           (points[:, 2] > -3) & (points[:, 2] < 1)
    points = points[mask]

    # 2. INCREASE RESOLUTION HERE
    # figsize=(width_inches, height_inches)
    # dpi=dots_per_inch
    # Total Pixels = (20*300) x (15*300) = 6000 x 4500 pixels
    fig = plt.figure(figsize=(20, 15), dpi=300) 
    
    ax = fig.add_subplot(111, projection='3d')

    # Styling
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # 3. Color Mapping
    dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    colors = plt.cm.turbo(dist / 40.0) 

    # 4. Scatter Plot
    # Adjusted 's' (point size). 
    # At high resolution, points look smaller. 
    # Increased from 0.5 to 1.5 to ensure they are visible.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               s=1.5, c=colors, marker='.', alpha=0.8, linewidth=0)

    # 5. Camera View
    ax.view_init(elev=30, azim=-160)

    # 6. Force Aspect Ratio
    max_range = np.array([points[:, 0].max()-points[:, 0].min(), 
                          points[:, 1].max()-points[:, 1].min(), 
                          points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 7. Save
    print("Render in progress (this may take a moment due to high DPI)...")
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Saved High-Res visualization to {output_path}")

def visualize_nuscenes_highres(bin_path, output_path):
    # 1. Load Data
    # CRITICAL CHANGE: nuScenes points are 5-dim [x, y, z, intensity, ring_index]
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)
    
    # We only need the first 4 columns for visualization (x,y,z,intensity)
    points = points[:, :4]

    # Downsampling (NuScenes is dense, so [::2] or [::3] is good)
    # points = points[::2]

    # Filter area (NuScenes LIDAR is 360 degrees, unlike KITTI's front-facing)
    # Let's visualize a 40x40m box around the car
    mask = (np.abs(points[:, 0]) < 40) & \
           (np.abs(points[:, 1]) < 40) & \
           (points[:, 2] > -3) & (points[:, 2] < 3)
    points = points[mask]

    # 2. Create Plot
    fig = plt.figure(figsize=(10, 10), dpi=300) 
    ax = fig.add_subplot(111, projection='3d')

    # Styling
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(False)
    ax.set_axis_off()
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # 3. Color Mapping
    dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    colors = plt.cm.turbo(dist / 40.0) 

    # 4. Scatter Plot
    # Adjusted 's' (point size). 
    # At high resolution, points look smaller. 
    # Increased from 0.5 to 1.5 to ensure they are visible.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               s=1.0, c=colors, marker='.', alpha=0.8, linewidth=0)

    # 5. Camera View
    ax.view_init(elev=30, azim=-70)

    # 6. Force Aspect Ratio
    max_range = np.array([points[:, 0].max()-points[:, 0].min(), 
                          points[:, 1].max()-points[:, 1].min(), 
                          points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 7. Save
    print("Render in progress...")
    plt.tight_layout()
    plt.savefig(output_path, facecolor='black', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Saved NuScenes visualization to {output_path}")


if __name__ == "__main__":
    # Replace with your path
    bin_file_kitti = "/DATA/kitti_mmdet3d/training/velodyne/000021.bin"
    bin_file_nuscenes = "/DATA/nuScenes/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915250396770.pcd.bin"
    visualize_kitti(bin_file_kitti, "kitti_occluded.png")
    # visualize_nuscenes_highres(bin_file_nuscenes, "nuscenes_scene.png")