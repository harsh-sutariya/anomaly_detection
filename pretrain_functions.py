import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import torch.optim as optim
import wandb

wandb.init(
    project="anomaly",

    config={
    "architecture": "Teacher-Decoder",
    "dataset": "ModelNet10"
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

def chamfer_distance(pc1, pc2):

    pc2_reshaped = pc2.view(-1, pc2.size(1) * pc2.size(2), pc2.size(3))
    pc1_expand = pc1.unsqueeze(2)
    pc2_expand = pc2_reshaped.unsqueeze(0)
    dist = torch.sum((pc1_expand - pc2_expand) ** 2, dim=-1)
    min_dist1, _ = torch.min(dist, dim=2)
    min_dist2, _ = torch.min(dist, dim=1)
    chamfer_dist = torch.mean(min_dist1, dim=1) + torch.mean(min_dist2, dim=1)
    return chamfer_dist.mean()

def pretrain_teacher_decoder(teacher, decoder, train_loader, val_loader, device, num_epochs=250, lr=1e-3, weight_decay=1e-6, save_path_teacher='best_teacher.pth', save_path_decoder='best_decoder.pth'):
    
    wandb.config.update({"num_epochs": num_epochs, "learning_rate": lr, "weight_decay": weight_decay})
    
    teacher.train()
    decoder.train()
    optimizer_teacher = optim.Adam(teacher.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    loss_values = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_points, _ in train_loader:
            batch_points = batch_points.to(device)
            
            optimizer_teacher.zero_grad()
            optimizer_decoder.zero_grad()

            features = teacher(batch_points)
            sampled_points = features[torch.randperm(features.size(0))[:batch_points.size(0)]]

            reconstructed_points = decoder(sampled_points)
            
            knn_idx = knn(batch_points, 8) 
            receptive_fields = compute_receptive_fields(batch_points, knn_idx, 1024)  

            receptive_fields_tensor = torch.stack(receptive_fields).to(device)
            loss = chamfer_distance(reconstructed_points, receptive_fields_tensor)

            loss.backward()
            optimizer_teacher.step()
            optimizer_decoder.step()
            
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        loss_values.append(epoch_loss)
        print(f'Epoch {epoch + 1}, Combined Training Loss: {epoch_loss}')
        wandb.log({"Pretrain Training Loss": epoch_loss}, step=epoch + 1)

        val_loss = 0.0
        teacher.eval()
        decoder.eval()
        with torch.no_grad():
            for batch_points, _ in val_loader:
                batch_points = batch_points.to(device)
                
                features = teacher(batch_points)
                
                reconstructed_points = decoder(features)
                
                knn_idx = knn(batch_points, 32) 
                receptive_fields = compute_receptive_fields(batch_points, knn_idx, 1024)  

                receptive_fields_tensor = torch.stack(receptive_fields).to(device)
                
                val_loss += chamfer_distance(reconstructed_points, receptive_fields_tensor).item()

        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}, Combined Validation Loss: {val_loss}')
        wandb.log({"Pretrain Validation Loss": val_loss}, step=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_teacher_model = teacher.state_dict()
            best_decoder_model = decoder.state_dict()
            torch.save(best_teacher_model, save_path_teacher)
            torch.save(best_decoder_model, save_path_decoder)
            print(f'Best models saved with validation loss: {val_loss}')

        teacher.train()
        decoder.train()

    wandb.finish()
    plt.plot(range(1, num_epochs + 1), loss_values, label='Combined Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Combined Model Training Curve')
    plt.legend()
    plt.show()

    if os.path.exists(save_path_teacher):
        teacher.load_state_dict(torch.load(save_path_teacher))
        print(f'Best teacher model loaded with validation loss: {best_val_loss}')
    else:
        print('No saved teacher model found.')

    if os.path.exists(save_path_decoder):
        decoder.load_state_dict(torch.load(save_path_decoder))
        print(f'Best decoder model loaded with validation loss: {best_val_loss}')
    else:
        print('No saved decoder model found.')

    print(f'Best Combined Validation Loss: {best_val_loss}')
