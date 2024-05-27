from student_data import get_loader
from torch.utils.data import DataLoader
import torch
import pickle
import models
from student_functions import train_student


train_loader, val_loader = get_loader()

pretrain_train_loader_file_path = "student_train_loader1024.pkl"
pretrain_val_loader_file_path = "student_val_loader1024.pkl"
teacher_model_weights_path = "best_teacher0.pth"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

with open(pretrain_train_loader_file_path, 'wb') as f:
    pickle.dump((train_loader), f)

with open(pretrain_val_loader_file_path, 'wb') as f:
    pickle.dump((val_loader), f)

with open(pretrain_train_loader_file_path, 'rb') as f:
    train_loader = pickle.load(f)

with open(pretrain_val_loader_file_path, 'rb') as f:
    val_loader = pickle.load(f)

teacher_model = models.TeacherModel(d=64, k=8).to(device)
student_model = models.TeacherModel(d=64, k=8).to(device)

train_student(teacher_model, student_model, teacher_model_weights_path, train_loader, val_loader, device)