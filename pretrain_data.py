import os
import urllib.request
import zipfile
import trimesh
from trimesh import transformations
import random
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset, DataLoader
from pretrain_functions import knn

def download_modelnet10(destination_dir):
    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    local_zip_path = os.path.join(destination_dir, "ModelNet10.zip")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if not os.path.exists(local_zip_path):
        print("Downloading ModelNet10 dataset...")
        urllib.request.urlretrieve(url, local_zip_path)
        print("Download complete.")

    print("Unzipping the dataset...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)
    print("Unzipping complete.")

def mesh_to_point_cloud(mesh, num_points=16000):
    point_cloud = mesh.sample(num_points)
    return point_cloud


def preprocess_modelnet10(download=True, num_points=16000):
    if download:
        download_modelnet10(destination_dir)    
    destination_dir = "./modelnet10"
    dataset_dir = os.path.join(destination_dir, "ModelNet10")
    
    point_clouds = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.off'):
                file_path = os.path.join(root, file)
                mesh = trimesh.load(file_path, force='mesh')
                if mesh.is_empty or mesh.extents is None:
                    continue
                point_cloud = mesh_to_point_cloud(mesh, num_points)
                point_clouds.append(point_cloud)

    return point_clouds


def generate_synthetic_scene(point_clouds, num_objects=10, scene_range=3, num_points=16000):
    scene_points = []

    for _ in range(num_objects):
        pc = random.choice(point_clouds)

        scale_factor = 1.0 / np.max(np.linalg.norm(pc, axis=1))
        pc *= scale_factor

        angles = np.random.uniform(0, 2 * np.pi, 3)
        rotation_matrix = transformations.euler_matrix(angles[0], angles[1], angles[2])[:3, :3]
        pc = np.dot(pc, rotation_matrix)

        translation = np.random.uniform(-scene_range, scene_range, 3)
        pc += translation

        scene_points.append(pc)

    scene_points = np.concatenate(scene_points, axis=0)
    print(f"Combined scene points shape: {scene_points.shape}")

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd = scene_pcd.farthest_point_down_sample(num_samples=1024)

    print(f"Final scene point cloud shape: {np.asarray(scene_pcd.points).shape}")
    return np.asarray(scene_pcd.points)

def generate_synthetic_dataset(point_clouds, num_scenes=525, num_train=500, num_points=16000):
    scenes = [generate_synthetic_scene(point_clouds, num_points=num_points) for _ in range(num_scenes)]
    print(f"Generated {len(scenes)} scenes")
    print(f"Shape of first scene: {scenes[0].shape if scenes else 'No scenes generated'}")
    return scenes[:num_train], scenes[num_train:]


class PointCloudDataset(Dataset):
    def __init__(self, point_clouds):
        self.point_clouds = point_clouds
        print(f"PointCloudDataset initialized with {len(point_clouds)} point clouds")

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        return torch.tensor(point_cloud, dtype=torch.float32), 0
    

def compute_average_neighbor_distance(point_clouds, k=8):
    distances = []
    for pc in point_clouds:
        pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0)
        idx = knn(pc_tensor, k)
        pc_tensor = pc_tensor.squeeze(0)
        for i in range(pc_tensor.shape[0]):
            neighbors = pc_tensor[idx[0, i, :]]
            neighbor_distances = torch.norm(pc_tensor[i] - neighbors, dim=1)
            distances.extend(neighbor_distances.tolist())
    average_distance = sum(distances) / len(distances)
    return average_distance

def normalize_point_clouds(point_clouds, average_distance):
    normalized_point_clouds = []
    for pc in point_clouds:
        normalized_pc = pc / average_distance
        normalized_point_clouds.append(normalized_pc)
    return normalized_point_clouds

def preprocess_and_normalize_modelnet10(num_points=16000, k=8):
    destination_dir = "./modelnet10"
    point_clouds = preprocess_modelnet10(num_points=num_points)
    average_distance = compute_average_neighbor_distance(point_clouds, k=k)
    normalized_point_clouds = normalize_point_clouds(point_clouds, average_distance)
    return normalized_point_clouds

def get_loader(download=True, normalize_data=True):
    point_clouds = preprocess_modelnet10(download=download)
    print(f"Processed {len(point_clouds)} point clouds.")
    print(f"Shape of each point cloud: {point_clouds[0].shape if point_clouds else 'No point clouds processed'}")

    if normalize_data:
        average_distance = compute_average_neighbor_distance(point_clouds)
        point_clouds = normalize_point_clouds(point_clouds, average_distance)
        print("Data normalized.")

    train_scenes, val_scenes = generate_synthetic_dataset(point_clouds)
    print(f"Generated {len(train_scenes)} training scenes and {len(val_scenes)} validation scenes.")
    print(f"Shape of a training scene: {train_scenes[0].shape if train_scenes else 'No training scenes'}")
    print(f"Shape of a validation scene: {val_scenes[0].shape if val_scenes else 'No validation scenes'}")

    train_data = PointCloudDataset(train_scenes)
    val_data = PointCloudDataset(val_scenes)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")

    return train_loader, val_loader
