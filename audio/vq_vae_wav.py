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

if __name__ == '__main__':
    df = pd.read_csv('../../dataset/ESC-50-master/meta/esc50.csv')
    file_path = '../../dataset/ESC-50-master/audio/'
    train_df, test_df = train_test_split(df, test_size = 0.2)
    train_dataset = ESC_dataset(train_df, file_path)
    test_dataset = ESC_dataset(test_df, file_path)
    train_variance = []
    for wav in train_dataset:
        train_variance.append(wav)
    train_variances = torch.var(torch.stack(train_variance))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = params.batch_size, shuffle = True, num_workers = 2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = params.batch_size, shuffle = True, num_workers = 2)
    model, optimizer = model.get_model(train_variances)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(10):
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
        os.makedirs('./model', exist_ok = True)
        torch.save(model.state_dict(), './model/model.pth')