import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import ResidualVQ, VectorQuantize
import random

from typing import Tuple, List, Optional
import params as params

#活性化関数の定義
elu = nn.ELU()
leaky = nn.LeakyReLU(0.3)

#残差ブロ�?ク
class Residual(nn.Module):
    def __init__(self, in_channels, dilation):
        super().__init__()
        self.conv_one = nn.Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = 7, stride = 1, dilation = dilation, padding = 'same')
        self.conv_two = nn.Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = 1, stride = 1)

    def forward(self, x):
        h = self.conv_one(x)
        h = elu(h)
        h = self.conv_two(h)
        return elu(x + h)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers = params.residual_layer_num):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([Residual(in_channels, params.dilations[i]) for i in range(num_residual_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Encoder_block(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.layer = nn.Conv1d(in_channels = in_channels, out_channels = in_channels * 2, kernel_size = 2 * stride, stride = stride , padding = (2 * stride) // 2)
        self.residual_stack = ResidualStack(in_channels * 2)
    
    def forward(self, x):
        x = self.layer(x)
        x = elu(x)
        return self.residual_stack(x)
        
class Decoder_block(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.layer = nn.ConvTranspose1d(in_channels = in_channels, out_channels = in_channels // 2, kernel_size = 2 * stride, stride = stride, padding = (stride // 2))
        self.residual_stack = ResidualStack(in_channels // 2)
    
    def forward(self, x):
        x = self.layer(x)
        x = elu(x)
        return self.residual_stack(x)

#エンコーダー
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.layer_one = nn.Conv1d(in_channels = in_channels, out_channels = hidden_dim // 16, kernel_size = 7, padding = 'same')
        self.layer_two = Encoder_block(in_channels = hidden_dim // 16, stride = 2)
        self.layer_three = Encoder_block(in_channels = hidden_dim // 8, stride = 3)
        self.layer_four = Encoder_block(in_channels = hidden_dim // 4, stride = 4)
        self.layer_five = Encoder_block(in_channels = hidden_dim // 2, stride = 5)
        
        self.pre_conv = nn.Conv1d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 3, padding = 'same')
        self.layer_list = [self.layer_one, self.layer_two, self.layer_three, self.layer_four, self.layer_five]

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        x = self.pre_conv(x)
        x = elu(x)
        return x

#�?コーダー
class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.pre_transconv = nn.Conv1d(in_channels, hidden_dim, kernel_size = 3, padding = 'same')
        self.layer_one = Decoder_block(in_channels = hidden_dim , stride = 5)
        self.layer_two = Decoder_block(in_channels = hidden_dim // 2, stride = 4)
        self.layer_three = Decoder_block(in_channels = hidden_dim // 4, stride = 3)
        self.layer_four = Decoder_block(in_channels = hidden_dim // 8, stride = 2)
        self.decoder_layer = [self.layer_one, self.layer_two, self.layer_three, self.layer_four]
        self.layer_five = nn.Conv1d(in_channels = hidden_dim // 16, out_channels = 1, kernel_size = 7, padding = 'same')

    def forward(self, x):
        x = elu(self.pre_transconv(x))
        for layer in self.decoder_layer:
            x = layer(x)
        return torch.tanh(self.layer_five(x))

class ReconstructionLoss(nn.Module):
    """Reconstruction loss from https://arxiv.org/pdf/2107.03312.pdf
    but uses STFT instead of mel-spectrogram
    """
    def __init__(self, eps = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input, target):
        loss = 0
        input = input.to(torch.float32)
        target = target.to(torch.float32)
        for i in range(6, 12):
            s = 2 ** i
            alpha = (s / 2) ** 0.5
            # We use STFT instead of 64-bin mel-spectrogram as n_fft=64 is too small
            # for 64 bins.
            x = torch.stft(input, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            x = torch.abs(x)
            y = torch.stft(target, n_fft=s, hop_length=s // 4, win_length=s, normalized=True, onesided=True, return_complex=True)
            y = torch.abs(y)
            if x.shape[-1] > y.shape[-1]:
                x = x[:, :, :y.shape[-1]]
            elif x.shape[-1] < y.shape[-1]:
                y = y[:, :, :x.shape[-1]]
            loss += torch.mean(torch.abs(x - y))
            loss += alpha * torch.mean(torch.square(torch.log(x + self.eps) - torch.log(y + self.eps)))
        return loss / (12 - 6)

#VQ-VAE
class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq, data_variance = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self.data_variance = data_variance
        self.recon_loss = ReconstructionLoss()

    def forward(self, x):
        z = self.encoder(x)
        z_permute = z.permute(0, 2, 1).contiguous()
        vq_quantize, embedding_idx, vq_loss = self.vq(z_permute)
        vq_quantize_reshape = vq_quantize.permute(0, 2, 1).contiguous()
        output = self.decoder(vq_quantize_reshape)
        if self.data_variance:
            recon_loss = self.recon_loss(output[:, 0, :], x [:, 0, :])
            vq_loss = vq_loss.mean()
            loss = recon_loss + vq_loss
            return {'z': z, 'x': x, 'loss': loss, 'recon_loss': recon_loss, 'vq_loss': vq_loss, 'vq_output': vq_quantize, 'output': output}
        else:
            return {'z': z, 'x': x, 'vq_output': vq_quantize, 'output': output}

class Resnet2d(nn.Module):
    def __init__(self, n_channels, factor, stride):
        super().__init__()
        self.layer_one = nn.Conv2d(in_channels = n_channels, out_channels = n_channels, kernel_size = (3, 3), padding = 'same')
        self.batch_norm_one = nn.BatchNorm2d(n_channels)
        self.layer_two = nn.Conv2d(in_channels = n_channels, out_channels = factor * n_channels, kernel_size = (stride[0] + 2, stride[1] + 2), stride = stride)
        self.layer_three = nn.Conv2d(in_channels = n_channels, out_channels = factor * n_channels, kernel_size = 1, stride = stride)
        self.batch_norm_two = nn.BatchNorm2d(factor * n_channels)
        self.pad = nn.ReflectionPad2d([(stride[1] + 1) // 2, (stride[1] + 2) // 2, (stride[0] + 1) // 2, (stride[0] + 2) // 2])

    def forward(self, input):
        x = self.layer_one(input)
        x = self.batch_norm_one(x)
        x = leaky(x)
        x = self.pad(x)
        x = self.layer_two(x)
        x = self.batch_norm_two(x)

        y = self.layer_three(input)
        y = self.batch_norm_two(y)
        return leaky(x + y)

class WaveDiscriminator(nn.Module):
    def __init__(self, n_channels ,resolution = 1):
        super().__init__()
        if resolution == 1:
            self.avg_pool = nn.Identity()
        else:
            self.abg_pool = nn.AbgPool2d(kernel_size = (resolution * 2), stride = resolution)
        self.layers = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(1, n_channels, kernel_size=15, padding=7)),
            nn.utils.weight_norm(nn.Conv1d(n_channels, 4 * n_channels, kernel_size=41, stride=4, padding=20, groups=4)),
            nn.utils.weight_norm(nn.Conv1d(4 * n_channels, 16 * n_channels, kernel_size=41, stride=4, padding=20, groups=16)),
            nn.utils.weight_norm(nn.Conv1d(16 * n_channels, 64 * n_channels, kernel_size=41, stride=4, padding=20, groups=64)),
            nn.utils.weight_norm(nn.Conv1d(64 * n_channels, 256 * n_channels, kernel_size=41, stride=4, padding=20, groups=256)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 256 * n_channels, kernel_size=5, padding=2)),
            nn.utils.weight_norm(nn.Conv1d(256 * n_channels, 1, kernel_size=3, padding=1)),
        ])

    def forward(self, x):
        x = self.avg_pool(x)
        features = []
        for layer in self.layers[:-1]:
            x = layer(x)
            features.append(x)
            x = leaky(x)
        features.append(self.layers[-1](x))
        return features

class STFTDiscriminator(nn.Module):
    def __init__(self, n_fft = 1024, hop_length = 256, n_channels = 32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n = n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1
        self.layers = nn.Sequential(
            nn.Conv2d(1, n_channels, kernel_size=7, padding='same'),
            nn.LeakyReLU(0.3, inplace=True),
            Resnet2d(n_channels, 2, stride=(2, 1)),
            Resnet2d(2 * n_channels, 2, stride=(2, 2)),
            Resnet2d(4 * n_channels, 1, stride=(2, 1)),
            Resnet2d(4 * n_channels, 2, stride=(2, 2)),
            Resnet2d(8 * n_channels, 1, stride=(2, 1)),
            Resnet2d(8 * n_channels, 2, stride=(2, 2)),
            nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))
        )

    def forward(self, x):
        x = torch.squeeze(input, 1).to(torch.float32)
        x = torch.stft(x, self.n_fft, self.hop_length, normalized = True, onesided = True, return_complex = True)
        x = torch.abs(x)
        x = x.unsqueeze(1)
        return self.layers(x)

#モ�?ルの読み込み
def get_model(data_variance = None):
    encoder = Encoder(in_channels = params.in_channels, hidden_dim = params.hidden_dim)
    decoder = Decoder(in_channels = params.embedding_dim, hidden_dim = params.hidden_dim)
    #vq = VectorQuantize(dim = params.embedding_dim, codebook_size = params.num_embeddings)
    vq = ResidualVQ(dim = params.embedding_dim, num_quantizers = 8, codebook_size = params.codebook_size, kmeans_init = True, kmeans_iters = 100, threshold_ema_dead_code = 2)
    model = VQVAE(encoder, decoder, vq, data_variance = data_variance)
    optimizer = torch.optim.AdamW(model.parameters(), lr = params.learning_rate)
    return model, optimizer