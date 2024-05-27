import torch
import open3d as o3d
import numpy as np
import models
import pickle
from skimage import io
import os
import matplotlib.pyplot as plt

def load_model(model, model_path, device):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print(f'Model loaded from {model_path}')
    else:
        raise FileNotFoundError(f'Model weights file not found at {model_path}')
    
def calculate_mu_sigma(teacher, data_loader, device):
    all_features = []
    teacher.eval()
    with torch.no_grad():
        for points in data_loader:
            points = points.to(device)
            features = teacher(points)
            all_features.append(features)
    all_features = torch.cat(all_features, dim=0)
    mu = all_features.mean(dim=0)
    sigma = all_features.std(dim=0)
    return mu, sigma

def inference_with_decoder(teacher, student, decoder, point_cloud, mu, sigma, device):
    teacher.eval()
    student.eval()
    decoder.eval()
    point_cloud = point_cloud.to(device).unsqueeze(0)

    with torch.no_grad():
        teacher_features = teacher(point_cloud)
        student_features = student(point_cloud)

        norm_teacher_features = (teacher_features - mu) / sigma
        errors = torch.norm(student_features - norm_teacher_features, dim=-1)
        errors = torch.norm(student_features - teacher_features, dim=-1)

        reconstructed_point_cloud = decoder(student_features)
        teacher_point_cloud = decoder(teacher_features)

    return errors.squeeze(0).cpu().numpy(), reconstructed_point_cloud.squeeze(0).cpu().numpy(), teacher_point_cloud.squeeze(0).cpu().numpy()

def visualize_point_cloud(point_cloud, title='Point Cloud'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd], window_name=title)

def get_loader(path):
    with open(path, 'rb') as f:
        train_loader = pickle.load(f)
    return train_loader

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

def normalize_point_cloud(point_cloud):
    average_distance = 15.199234631065725
    return point_cloud / average_distance

def get_point_cloud_from_tiff(tiff_path):
    num_points = 1024
    point_cloud = io.imread(tiff_path)
    point_cloud = point_cloud.reshape(-1, 3)
    if len(point_cloud) > num_points:
            indices = farthest_point_sampling(point_cloud, num_points)
            point_cloud = point_cloud[indices]
    normalized_point_cloud=normalize_point_cloud(point_cloud)
    return normalized_point_cloud
    
def visualize_point_cloud_anomaly(point_cloud, anomaly_scores, title='Point Cloud with Anomalies', threshold=0.873):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    anomaly_scores_normalized = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    
    colors = np.zeros((anomaly_scores_normalized.shape[0], 3))
    colors[anomaly_scores_normalized > threshold] = [1, 1, 0]
    colors[anomaly_scores_normalized <= threshold] = [0, 0, 0]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name=title)

if __name__ == "__main__":
    teacher_model_path = 'best_teacher0.pth'
    student_model_path = 'best_student.pth'
    decoder_model_path = 'best_decoder0.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher = models.TeacherModel(d=64, k=8).to(device)
    student = models.TeacherModel(d=64, k=8).to(device)
    decoder = models.DecoderModel(d=64).to(device)

    load_model(teacher, teacher_model_path, device)
    load_model(student, student_model_path, device)
    load_model(decoder, decoder_model_path, device)

    train_loader = get_loader(path='student_train_loader1024.pkl')
    mu, sigma = calculate_mu_sigma(teacher, train_loader, device)

    # tiff_path = './mvtec_3d_anomaly_detection/cookie/test/crack/xyz/002.tiff' #0.9
    # tiff_path = './mvtec_3d_anomaly_detection/cable_gland/test/thread/xyz/001.tiff' # 0.903
    tiff_path = './mvtec_3d_anomaly_detection/peach/test/hole/xyz/010.tiff' # 0.873
    point_cloud=get_point_cloud_from_tiff(tiff_path)
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32)

    anomaly_scores, reconstructed_point_cloud, teacher_point_cloud = inference_with_decoder(teacher, student, decoder, point_cloud, mu, sigma, device)

    print("Anomaly scores for each point in the point cloud:", anomaly_scores)

    original_point_cloud_np = point_cloud.cpu().numpy()

    visualize_point_cloud(original_point_cloud_np, title='Original Point Cloud')
    # visualize_point_cloud(reconstructed_point_cloud, title='Student Point Cloud')
    # visualize_point_cloud(teacher_point_cloud, title='Teacher Point Cloud')
    visualize_point_cloud_anomaly(point_cloud, anomaly_scores)