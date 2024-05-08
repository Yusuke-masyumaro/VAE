import torch
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

import params as params
import model as model
import os
import wandb

epochs = 200

#WandBでログの取得
wandb.init(
    project = "VAE-ESC50",
    config = {
    'in_channels':params.in_channels,
    'hidden_dim':params.hidden_dim,
    'residual_hidden_dim':params.residual_hidden_dim,
    'num_residual_layers':params.num_residual_layers,
    'embedding_dim':params.embedding_dim,
    'num_embeddings':params.num_embeddings,
    'learning_rate':params.learning_rate,
    'batch_size':params.batch_size,
    "architecture": "VAE",
    "dataset": "ESC50",
    "epochs": epochs,
    }
)

#sampling rate 16Khz, 16bit, 3sec
class Dataset_16K(Dataset):
    def __init__(self, wav_list, path):
        self.path = path
        self.data_list = []
        for file_name in tqdm(wav_list):
            wav, _ = torchaudio.load(self.path + file_name)
            self.data_list.append(wav)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

if __name__ == '__main__':
    
    #データセットの読み込み、データローダの作成
    path_list = ['../dataset/ESC-50-master_16K/', '../dataset/Urban_16K/'] 
    train_dataset_list = []
    test_dataset_list = []
    for i in path_list:
        train_list, test_list = train_test_split(os.listdir(i), test_size = 0.2)
        train_data = Dataset_16K(train_list, i)
        test_data = Dataset_16K(test_list, i)
        train_dataset_list.append(train_data)
        test_dataset_list.append(test_data)
    train_dataset = ConcatDataset(train_dataset_list)
    test_dataset = ConcatDataset(test_dataset_list)
    print('train: {}, test: {}'.format(len(train_dataset), len(test_dataset)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params.batch_size, shuffle = True, num_workers = 2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params.batch_size, shuffle = True, num_workers = 2)
    
    #訓練データの分散を計算
    train_variance = []
    for wav in train_dataset:
        train_variance.append(wav)
    train_variances = torch.var(torch.stack(train_variance))
    
    #モデルの読み込み
    model, optimizer = model.get_model(train_variances)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    #エポックごとにpthファイルを保存
    for epoch in range(epochs):
        train_loss = 0.0
        train_recon_loss = 0.0
        train_vq_loss = 0.0
        model.train()
        for _, data in tqdm(enumerate(train_loader)):
            x = data.to(device)
            output = model(x)
            optimizer.zero_grad()
            loss = output['loss']
            loss.backward()
            train_loss += loss.item()
            train_recon_loss += output['recon_loss'].item()
            train_vq_loss += output['vq_loss'].item()
            optimizer.step()
            
        model.eval()
        test_recon_loss = 0.0
        with torch.no_grad():
            for _, data in tqdm(enumerate(test_loader)):
                x = data.to(device)
                output = model(x)
                test_recon_loss += output['loss'].item()
        
        scheduler.step()

        print('epoch: {}, loss: {}, recon_loss: {}, vq_loss: {}'.format(epoch, train_loss / len(train_loader.dataset),train_recon_loss / len(train_loader.dataset), train_vq_loss / len(train_loader.dataset)))
        print('epoch: {}, loss: {}'.format(epoch, test_recon_loss / len(test_loader.dataset)))
        
        wandb.log({
            "train_loss": train_loss / len(train_loader.dataset),
            'train_recon_loss': train_recon_loss / len(train_loader.dataset),
            'train_vq_loss': train_vq_loss / len(train_loader.dataset),
            "test_loss": test_recon_loss / len(test_loader.dataset)
        })
        
        os.makedirs('./model', exist_ok = True)
        torch.save(model.state_dict(), './model/model.pth')