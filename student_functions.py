import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import wandb

wandb.init(
    project="anomaly_student_training",
    config={
        "architecture": "Student-Decoder",
        "dataset": "MVTec 3D-AD 1/8"
    }
)

def compute_receptive_fields(points, knn_idx, L):
    B, N, _ = points.shape
    receptive_fields = []
    for i in range(N):
        rf = set([i])
        for l in range(1, L + 1):
            next_hop = set()
            for q_idx in rf:
                if isinstance(q_idx, int) and q_idx >= N:
                    continue
                knn_q = knn_idx[0, q_idx, :]
                knn_q = knn_q.long()
                next_hop.update(knn_q.tolist())
            rf.update(next_hop)
        rf = [idx for idx in rf if isinstance(idx, int) and idx < N]
        receptive_fields.append(points[:, rf, :])
    max_rf_size = max(len(rf) for rf in receptive_fields)
    padded_fields = [F.pad(rf, (0, 0, 0, max_rf_size - rf.size(1), 0, 0)) for rf in receptive_fields]
    return padded_fields

def knn(x, k):
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    pairwise_distance = xx + xx.transpose(2, 1) - 2 * torch.matmul(x, x.transpose(2, 1))
    idx = pairwise_distance.topk(k=k, dim=-1, largest=False)[1]
    return idx

def feature_normalization(teacher_features, mu, sigma):
    return (teacher_features - mu) / sigma

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

def train_student(student, teacher, teacher_model_weights_path, train_loader, val_loader, device, num_epochs=100, lr=1e-3, weight_decay=1e-5, save_path_student='best_student.pth'):
    
    if os.path.exists(teacher_model_weights_path):
        teacher.load_state_dict(torch.load(teacher_model_weights_path))
        print(f'Pretrained teacher model loaded from {teacher_model_weights_path}')
    else:
        raise FileNotFoundError(f'Teacher weights file not found at {teacher_model_weights_path}')
    
    student.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    wandb.config.update({"num_epochs": num_epochs, "learning_rate": lr, "weight_decay": weight_decay})
    
    teacher.eval()
    student.train()
    optimizer_student = optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    loss_values = []

    mu, sigma = calculate_mu_sigma(teacher, train_loader, device)
    sigma_inv = 1.0 / sigma

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_points in train_loader:
            batch_points = batch_points.to(device)
            
            optimizer_student.zero_grad()

            teacher_features = teacher(batch_points).detach()
            student_features = student(batch_points)
            
            norm_teacher_features = (teacher_features - mu) * sigma_inv
            loss = F.mse_loss(student_features, norm_teacher_features)

            loss.backward()
            optimizer_student.step()
            
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        loss_values.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Student Training Loss: {epoch_loss}')
        wandb.log({"Student Training Loss": epoch_loss}, step=epoch + 1)

        val_loss = 0.0
        student.eval()
        with torch.no_grad():
            for batch_points in val_loader:
                batch_points = batch_points.to(device)
                
                teacher_features = teacher(batch_points)
                student_features = student(batch_points)
                
                norm_teacher_features = (teacher_features - mu) * sigma_inv
                val_loss += F.mse_loss(student_features, norm_teacher_features).item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Student Validation Loss: {val_loss}')
        wandb.log({"Student Validation Loss": val_loss}, step=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_student_model = student.state_dict()
            torch.save(best_student_model, save_path_student)
            print(f'Best student model saved with validation loss: {val_loss}')

        student.train()

    plt.plot(range(1, num_epochs + 1), loss_values, label='Student Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Student Model Training Curve')
    plt.legend()
    plt.show()

    if os.path.exists(save_path_student):
        student.load_state_dict(torch.load(save_path_student))
        print(f'Best student model loaded with validation loss: {best_val_loss}')
    else:
        print('No saved student model found.')

    print(f'Best Student Validation Loss: {best_val_loss}')
