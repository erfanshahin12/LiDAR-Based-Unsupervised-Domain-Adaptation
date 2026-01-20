import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import math

def get_box_corners(line):
    """
    Parses a label line and calculates the 8 corners of the 3D box 
    in LiDAR coordinates.
    """
    data = line.strip().split()
    obj_class = data[0]

    # Skip 'DontCare'
    if obj_class == 'DontCare':
        return None, None
    
    # 1. Parse Dimensions and Location (Camera Coords)
    h, w, l = float(data[8]), float(data[9]), float(data[10])
    tx, ty, tz = float(data[11]), float(data[12]), float(data[13])
    ry = float(data[14]) # Rotation around Y-axis (Camera coords)

    # 2. Convert Camera Coords to LiDAR Coords (Approximate)
    # Standard KITTI conversion (swapping axes)
    # Camera: x (right), y (down), z (forward)
    # LiDAR:  x (forward), y (left), z (up)
    
    # Position
    # x_lidar = z_cam
    # y_lidar = -x_cam
    # z_lidar = -y_cam (shifted up by h/2 because label is at bottom face)
    x_lidar = tz + 0.27 # 0.27 is a standard calib offset
    y_lidar = -tx
    z_lidar = -ty + (h / 2) # Move from bottom-center to geometric center

    # Rotation (LiDAR rotates around Z, Camera around Y)
    # The zero-angle definition is also different
    rot_lidar = -(ry + np.pi / 2)

    # 3. Calculate 8 Corners relative to center
    # 3D Bounding Box with size (l, w, h)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]

    # 4. Rotate and Translate
    c, s = np.cos(rot_lidar), np.sin(rot_lidar)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d = np.dot(R, corners_3d)
    
    corners_3d[0, :] += x_lidar
    corners_3d[1, :] += y_lidar
    corners_3d[2, :] += z_lidar
    
    return corners_3d.T, obj_class

def visualize_occluded_object(bin_path, label_path, target_obj_idx):
    # 1. Load Point Cloud
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # points = points[::3] # Downsample

    # 2. Load Label and get the specific object box
    with open(label_path, 'r') as f:
        lines = f.readlines()
        target_line = lines[target_obj_idx]
        
    corners = get_box_corners(target_line)
    
    # 3. Setup Plot
    fig = plt.figure(figsize=(20, 15), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_axis_off()
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # 4. Filter Points to focus on the object
    # We define a generic ROI around the box center to zoom in
    mask = (points[:, 0] > 0) & (points[:, 0] < 40) & \
           (np.abs(points[:, 1]) < 20) & \
           (points[:, 2] > -3) & (points[:, 2] < 1)
    points = points[mask]
    
    # 5. Plot Points
    dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    colors = plt.cm.turbo(dist / 40.0) 

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               s=1.5, c=colors, marker='.', alpha=0.8, linewidth=0)

    # 6. Plot Bounding Box Lines
    # The order of corners to draw lines:
    # Top face: 0-1, 1-2, 2-3, 3-0
    # Bottom face: 4-5, 5-6, 6-7, 7-4
    # Vertical pillars: 0-4, 1-5, 2-6, 3-7
    
    def draw_line(p1, p2, color='white'):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=0.5)

    # Draw the 12 lines
    for i in range(4):
        draw_line(corners[i], corners[(i+1)%4])           # Top Loop
        draw_line(corners[i+4], corners[(i+1)%4 + 4])     # Bottom Loop
        draw_line(corners[i], corners[i+4])               # Verticals

    # Draw "Front" indicator (Line between 0 and 4 is usually front-right in this logic)
    # Let's draw an X on the front face to identify heading
    # Front face in this corner mapping is usually 0-1-5-4
    # draw_line(corners[0], corners[5], color='yellow')
    # draw_line(corners[1], corners[4], color='yellow')

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
    
    plt.savefig("occluded_object_viz.png", facecolor='black', bbox_inches='tight')
    print("✅ Saved visualization to occluded_object_viz.png")

def visaulize_objects(bin_path, label_path):
    # 1. Load Point Cloud
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    
    # --- Load All Boxes ---
    boxes = []
    classes = []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            corners, cls = get_box_corners(line)
            if corners is not None:
                boxes.append(corners)
                classes.append(cls)
    
    # 3. Setup Plot
    fig = plt.figure(figsize=(20, 15), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_axis_off()
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    # 4. Filter Points to focus on the object
    # We define a generic ROI around the box center to zoom in
    mask = (points[:, 0] > 0) & (points[:, 0] < 60) & \
           (np.abs(points[:, 1]) < 30) & \
           (points[:, 2] > -3) & (points[:, 2] < 1)
    points = points[mask]
    
    # 5. Plot Points
    dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    colors = plt.cm.turbo(dist / 60.0) 

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               s=1.5, c=colors, marker='.', alpha=0.5, linewidth=0)

    class_colors = {
        'Car': 'red',
        'Van': 'orange',
        'Truck': 'yellow',
        'Pedestrian': 'cyan',
        'Person_sitting': 'cyan',
        'Cyclist': 'lime',
        'Tram': 'magenta',
        'Misc': 'white'
    }
    # 6. Plot Bounding Box Lines
    # The order of corners to draw lines:
    # Top face: 0-1, 1-2, 2-3, 3-0
    # Bottom face: 4-5, 5-6, 6-7, 7-4
    # Vertical pillars: 0-4, 1-5, 2-6, 3-7
    
    def draw_line(p1, p2, color):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, linewidth=0.5)

    print(f"Drawing {len(boxes)} boxes...")

    for corners, cls in zip(boxes, classes):
        color = class_colors.get(cls, 'white')
        
        # Check if box is in valid range (optional, prevents drawing boxes behind camera)
        if corners[:, 0].min() < 0: continue

        # Draw 12 edges
        for k in range(4):
            # Top Face Loop
            draw_line(corners[k], corners[(k+1)%4], color)
            # Bottom Face Loop
            draw_line(corners[k+4], corners[(k+1)%4 + 4], color)
            # Pillars
            draw_line(corners[k], corners[k+4], color)

        # Draw Front Indicator (Cross on front face)
        # Front face is usually indices 0-1-5-4 in this corner order
        # draw_line(corners[0], corners[5], color='yellow')
        # draw_line(corners[1], corners[4], color='yellow')

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
    
    plt.savefig("scene_bboxes.png", facecolor='black', bbox_inches='tight')
    print("✅ Saved visualization to scene_bboxes.png")

# Usage Example
# Replace 000000 with the file ID found in Script 1
bin_file = "/DATA/kitti_mmdet3d/training/velodyne/000412.bin" 
label_file = "/DATA/kitti_mmdet3d/training/label_2/000412.txt"
obj_index = 6 # The line number from Script 1

# visualize_occluded_object(bin_file, label_file, obj_index)
visaulize_objects(bin_file, label_file)