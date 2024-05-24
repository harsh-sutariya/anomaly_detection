from pretrain_data import get_loader
import models
from pretrain_functions import pretrain_teacher_decoder
from torch.utils.data import DataLoader
import torch
import pickle

train_loader, val_loader = get_loader(download=False, normalize_data=True)

pretrain_train_loader_file_path = "pretrain_train_loader1.pkl"
pretrain_val_loader_file_path = "pretrain_val_loader1.pkl"


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
decoder_model = models.DecoderModel(d=64).to(device)

pretrain_teacher_decoder(teacher_model, decoder_model, train_loader, val_loader, device)