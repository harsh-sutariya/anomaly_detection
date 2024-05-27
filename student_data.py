import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io


def load_point_clouds(folder_path, num_points=1024):
    tiff_files = sorted(os.listdir(folder_path))
    point_clouds = []
    for file in tiff_files:
        tiff_path = os.path.join(folder_path, file)
        point_cloud = io.imread(tiff_path)
        point_cloud = point_cloud.reshape(-1, 3)
        if len(point_cloud) > num_points:
            indices = farthest_point_sampling(point_cloud, num_points)
            point_cloud = point_cloud[indices]
        point_clouds.append(point_cloud)
    return point_clouds

def farthest_point_sampling(point_cloud, num_points):
    num_points_input = point_cloud.shape[0]
    center_index = np.random.choice(num_points_input, 1)
    center_point = point_cloud[center_index]
    distances = np.sum((point_cloud - center_point) ** 2, axis=1)
    sampled_indices = [center_index[0]]
    for _ in range(num_points - 1):
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        farthest_point = point_cloud[farthest_index]
        distances = np.minimum(distances, np.sum((point_cloud - farthest_point) ** 2, axis=1))
    return sampled_indices

def compute_average_distance(point_clouds):
    distances = []
    for pc in point_clouds:
        for i in range(pc.shape[0]):
            for j in range(i + 1, pc.shape[0]):
                distances.append(np.linalg.norm(pc[i] - pc[j]))
    return np.mean(distances)

def normalize_point_cloud(point_cloud, average_distance):
    return point_cloud / average_distance

class AnomalyDetectionDataset(Dataset):
    def __init__(self, data_folder, mode='train', num_points=1024):
        self.data_folder = data_folder
        self.mode = mode
        self.num_points = num_points
        self.classes = sorted(os.listdir(data_folder))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.point_clouds = self.load_data()

    def load_data(self):
        point_clouds = []
        normalized_point_clouds=[]
        for cls_name in self.classes:
            cls_folder = os.path.join(self.data_folder, cls_name, self.mode, 'good', 'xyz')
            for point_cloud in load_point_clouds(cls_folder, self.num_points):
                point_clouds.append(point_cloud)
        average_distance = compute_average_distance(point_clouds)
        for point_cloud in point_clouds:
            normalized_point_cloud = normalize_point_cloud(point_cloud, average_distance)
            normalized_point_clouds.append(normalized_point_cloud)
        return normalized_point_clouds

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        return torch.tensor(point_cloud, dtype=torch.float32)

def prepare_data_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_loader():
    data_folder = 'mvtec_3d_anomaly_detection_eighth'
    batch_size = 1
    train_dataset = AnomalyDetectionDataset(data_folder, mode='train')
    val_dataset = AnomalyDetectionDataset(data_folder, mode='validation')
    train_loader = prepare_data_loader(train_dataset, batch_size)
    val_loader = prepare_data_loader(val_dataset, batch_size)

    return train_loader, val_loader
