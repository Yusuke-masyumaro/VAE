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

epochs = 400

wandb.init(
    project = "VAE-ESC50",
    config = {
    'in_channels':params.in_channels,
    'hidden_dim':params.hidden_dim,
    'embedding_dim':params.embedding_dim,
    'codebook_size':params.codebook_size,
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
    
    path_list = ['../dataset/ESC-50-master_16K/', '../dataset/Urban_16K/', '../dataset/librispeech_16K/'] 
    train_dataset_list = []
    test_dataset_list = []
    for i in path_list:
        if i == '../dataset/librispeech_16K/':
            train_data = Dataset_16K(os.listdir(i + 'train-clean-100/'), i + 'train-clean-100/')
            test_data = Dataset_16K(os.listdir(i + 'test-clean/'), i + 'test-clean/')
            train_dataset_list.append(train_data)
            test_dataset_list.append(test_data)
        else:
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
    
    train_variance = []
    for wav in train_dataset:
        train_variance.append(wav)
    train_variances = torch.var(torch.stack(train_variance))
    
    model, optimizer_g, optimizer_d, wav_discriminator, stft_discriminator = model.get_model(train_variances)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs + 1):
        train_loss = 0.0
        train_recon_loss = 0.0
        train_vq_loss = 0.0
        model.train()
        for _, data in tqdm(enumerate(train_loader)):
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()
            x = data.to(device)
            output = model(x)

            stft_out = stft_discriminator(output['output'])
            g_stft_loss = torch.mean(torch.relu(1 - stft_out))
            g_wave_loss = 0
            g_feature_loss = 0
            for i in range(3):
                feature_input = wav_discriminator[i](output[x])
                feature_out = wav_discriminator[i](output['output'])
                g_wave_loss += torch.mean(torch.relu(1 - feature_out[-1]))
                g_feature_loss += sum(torch.mean(torch.abs(f1 - f2)) for f1, f2 in zip(feature_input[:-1], feature_out[:-1])) / (len(feature_input) - 1)

            g_feature_loss = g_feature_loss / 3
            g_adversarial_loss = (g_stft_loss + g_wave_loss) / 4
            g_loss = g_adversarial_loss + (100 * g_feature_loss) + output['loss']
            g_loss.backward()
            optimizer_g.step()

            loss = output['loss']
            loss.backward()
            train_loss += g_loss.item()
            train_recon_loss += output['recon_loss'].item()
            train_vq_loss += output['vq_loss'].item()

            #discriminator training
            d_stft_out = stft_discriminator(x)
            d_stft_loss = torch.mean(torch.relu(1 - d_stft_out))
            d_stft_out = stft_discriminator(output['output'])
            d_stft_loss += torch.mean(torch.relu(1 + d_stft_out))

            d_wave_loss = 0
            for i in range(3):
                feature_input = wav_discriminator[i](x)
                d_wave_loss += torch.mean(torch.relu(1 - feature_input))
                feature_out = wav_discriminator[i](output['output'])
                d_wave_loss += torch.mean(torch.relu(1 + feature_out))
            d_loss = (d_stft_loss + d_wave_loss) / 4
            d_loss.backward()
            optimizer_d.step()

        model.eval()
        test_recon_loss = 0.0
        with torch.no_grad():
            for _, data in tqdm(enumerate(test_loader)):
                x = data.to(device)
                output = model(x)
                test_recon_loss += output['loss'].item()


        print('epoch: {}, loss: {}, recon_loss: {}, vq_loss: {}'.format(epoch, train_loss / len(train_loader.dataset),train_recon_loss / len(train_loader.dataset), train_vq_loss / len(train_loader.dataset)))
        print('epoch: {}, test_recon_loss: {}'.format(epoch, test_recon_loss / len(test_loader.dataset)))

        wandb.log({
            "train_loss": train_loss / len(train_loader.dataset),
            'train_recon_loss': train_recon_loss / len(train_loader.dataset),
            'train_vq_loss': train_vq_loss / len(train_loader.dataset),
            "test_loss": test_recon_loss / len(test_loader.dataset)
        })

        if epoch % 20 == 0:
            os.makedirs('./model', exist_ok = True)
            torch.save(model.state_dict(), './model/model_{}.pth'.format(epoch))
