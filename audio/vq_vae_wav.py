import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import params as params
import model as model
import os
import wandb

epochs = 50

wandb.init(
    # set the wandb project where this run will be logged
    project="VAE-ESC50",

    # track hyperparameters and run metadata
    config={
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

class ESC_dataset(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.path = path
        self.data_list = []
        for file_name in tqdm(df['filename'].values):
            wav, sr = torchaudio.load(self.path + file_name)
            wav = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)(wav)
            self.data_list.append(wav)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.data_list[idx]

class ESC_dataset_16K(Dataset):
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
    '''
    df = pd.read_csv('../dataset/ESC-50-master/meta/esc50.csv')
    file_path = '../dataset/ESC-50-master/audio/'
    train_df, test_df = train_test_split(df, test_size = 0.2)
    train_dataset = ESC_dataset(train_df, file_path)
    test_dataset = ESC_dataset(test_df, file_path)
    '''
    
    file_path = '../dataset/ESC-50-master_16K/'
    data_list = os.listdir(file_path)
    train_list, test_list = train_test_split(data_list, test_size = 0.2)
    train_dataset = ESC_dataset_16K(train_list, file_path)
    test_dataset = ESC_dataset_16K(test_list, file_path)
    print('train: {}, test: {}'.format(len(train_dataset), len(test_dataset)))
    
    train_variance = []
    for wav in train_dataset:
        train_variance.append(wav)
    train_variances = torch.var(torch.stack(train_variance))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params.batch_size, shuffle = True, num_workers = 2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params.batch_size, shuffle = True, num_workers = 2)
    model, optimizer = model.get_model(train_variances)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(epochs):
        train_losses = 0.0
        train_recon_loss = 0.0
        for _, data in tqdm(enumerate(train_loader)):
            x = data.to(device)
            output = model(x)
            optimizer.zero_grad()

            loss = output['loss']
            loss.backward()
            train_losses += loss.item()
            train_recon_loss += output['reconstructed_error'].item()
            optimizer.step()

        model.eval()
        test_losses = 0.0
        test_recon_loss = 0.0
        with torch.no_grad():
            for _, data in tqdm(enumerate(test_loader)):
                x = data.to(device)
                output = model(x)
                loss = output['loss']
                test_losses += loss.item()
                test_recon_loss += output['reconstructed_error'].item()

        print('epoch: {}, loss: {}, recon_loss: {}'.format(epoch, train_losses / len(train_loader.dataset), train_recon_loss / len(train_loader.dataset)))
        print('epoch: {}, loss: {}, recon_loss: {}'.format(epoch, test_losses / len(test_loader.dataset), test_recon_loss / len(test_loader.dataset)))
        print('output: {}, x: {}'.format(output['output'].shape, x.shape))
        
        wandb.log({
            "train_loss": train_losses / len(train_loader.dataset),
            "train_recon_loss": train_recon_loss / len(train_loader.dataset),
            "test_loss": test_losses / len(test_loader.dataset),
            "test_recon_loss": test_recon_loss / len(test_loader.dataset)
        })
        
        os.makedirs('./model', exist_ok = True)
        torch.save(model.state_dict(), './model/model.pth')