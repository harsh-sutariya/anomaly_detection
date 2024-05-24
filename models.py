import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SharedMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.mlp(x)

class LocalFeatureAggregation(nn.Module):
    def __init__(self, d, k):
        super(LocalFeatureAggregation, self).__init__()
        self.shared_mlp = SharedMLP(6, d // 2)
        self.local_mlp = SharedMLP(d, d // 2)
        self.d = d
        self.k = k

    def forward(self, points, features, knn_idx):
        B, N, _ = points.shape
        _, K = knn_idx.shape
        
        knn_points = points.unsqueeze(2).expand(B, N, self.k, 3).gather(1, knn_idx.unsqueeze(-1).expand(B, N, self.k, 3))
        point_diff = points.unsqueeze(2).expand(B, N, self.k, 3) - knn_points 
        geometric_features = torch.cat([point_diff, point_diff.norm(dim=-1, keepdim=True)], dim=-1)
        geometric_features = geometric_features.view(B * N * self.k, -1)

        geometric_features = self.shared_mlp(geometric_features).view(B, N, self.k, self.d // 2)
        
        knn_features = features.unsqueeze(2).expand(B, N, self.k, -1).gather(1, knn_idx.unsqueeze(-1).expand(B, N, self.k, self.d))
        knn_features = torch.cat([knn_features, geometric_features], dim=-1).view(B * N * self.k, -1)
        
        aggregated_features = self.local_mlp(knn_features).view(B, N, self.k, self.d // 2)
        return aggregated_features.mean(dim=2)

class ResidualBlock(nn.Module):
    def __init__(self, d, k):
        super(ResidualBlock, self).__init__()
        self.shared_mlp = SharedMLP(d, d // 4)
        self.lfa1 = LocalFeatureAggregation(d // 4, d // 2)
        self.lfa2 = LocalFeatureAggregation(d // 2, d)
        self.shared_mlp = SharedMLP(d, d)

    def forward(self, points, features, knn_idx):
        x = features
        x = self.lfa1(points, x, knn_idx)
        x = self.lfa2(points, x, knn_idx)
        x = self.shared_mlp(x)
        return features + x

class TeacherModel(nn.Module):
    def __init__(self, d, k, num_residual_blocks=4):
        super(TeacherModel, self).__init__()
        self.shared_mlp = SharedMLP(3, d)
        self.residual_blocks = nn.ModuleList([ResidualBlock(d, k) for _ in range(num_residual_blocks)])
        self.final_mlp = nn.Linear(d, d)
        self.d = d

    def forward(self, points, knn_idx=None):
        features = self.shared_mlp(points)
        if knn_idx is not None:
            for block in self.residual_blocks:
                features = block(points, features, knn_idx)
        features = self.final_mlp(features)
        return features

class DecoderModel(nn.Module):
    def __init__(self, d=64, m=1024):
        super(DecoderModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(128, m)
        )

    def forward(self, features):
        reconstructed_points = self.mlp(features)
        return reconstructed_points