import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os

import params_image as params
import model_image as model

class Pokemon_dataset(Dataset):
    def __init__(self, df, file_path, transforms = None):
        self.df = df
        self.transforms = transforms
        self.image_list = []
        for name in tqdm(df['Name'].values):
            image = Image.open(file_path + name + '.jpg')
            self.image_list.append(image)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        if self.transforms:
            image = self.transforms(image)
        return image

if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.Resize((params.image_size, params.image_size)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    df = pd.read_csv('../../dataset/Pokemon/pokemon.csv')
    dataset = Pokemon_dataset(df, '../../dataset/Pokemon/images/', transform)
    train_data, test_data = train_test_split(dataset, test_size=0.2)
    train_variance = []
    for wav in train_data:
        train_variance.append(wav)
    train_variances = torch.var(torch.stack(train_variance))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = params.batch_size, shuffle = True, num_workers = 2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = params.batch_size, shuffle = True, num_workers = 2)
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
        print('output: {}, x: {}'.format(output['output'].shape, x.shape))
        os.makedirs('./model', exist_ok=True)
        torch.save(model.state_dict(), './model/model.pth')