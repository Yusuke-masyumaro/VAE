import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ, VectorQuantize

import params as params

#活性化関数の定義
relu = nn.ReLU()
leaky = nn.LeakyReLU(0.2)

#残差ブロック
class Residual(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_hidden):
        super().__init__()
        self.conv_one = nn.Conv1d(in_channels = in_channels, out_channels = num_residual_hidden, kernel_size = 3, stride = 1, padding = 1)
        self.conv_two = nn.Conv1d(in_channels = num_residual_hidden, out_channels = hidden_dim, kernel_size = 1, stride = 1)

    def forward(self, x):
        h = relu(x)
        h = relu(self.conv_one(h))
        h = self.conv_two(h)
        return x + h

class ResidualStack(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hidden):
        super().__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([Residual(in_channels, hidden_dim, num_residual_hidden) for _ in range(num_residual_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return relu(x)

#エンコーダー
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_residual_layers = num_residual_layers
        self.residual_hidden_dim = residual_hidden_dim

        self.e_layer_one = nn.Conv1d(in_channels = self.in_channels, out_channels = self.hidden_dim // 4, kernel_size = 5, stride = 2, padding = 2)
        self.e_layer_two = nn.Conv1d(in_channels = self.hidden_dim // 4, out_channels = self.hidden_dim // 2, kernel_size = 5, stride = 2, padding = 2)
        self.e_layer_three = nn.Conv1d(in_channels = self.hidden_dim // 2, out_channels = self.hidden_dim, kernel_size = 5, stride = 2, padding = 2)
        self.e_layer_four = nn.Conv1d(in_channels = self.hidden_dim, out_channels = self.hidden_dim, kernel_size = 5, stride = 2, padding = 2)
        self.residual_stack = ResidualStack(self.hidden_dim, self.hidden_dim, self.num_residual_layers, self.residual_hidden_dim)

        self.encoder_layer = nn.ModuleList([self.e_layer_one, self.e_layer_two, self.e_layer_three, self.e_layer_four])

    def forward(self, x):
        for layer in self.encoder_layer:
            if layer != self.e_layer_four:
                x = layer(x)
                x = leaky(x)
            else:
                x = layer(x)
        return self.residual_stack(x)

#デコーダー
class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_residual_layers = num_residual_layers
        self.residual_hidden_dim = residual_hidden_dim

        self.d_layer_one = nn.ConvTranspose1d(in_channels = self.in_channels, out_channels = self.hidden_dim, kernel_size = 4, stride = 2, padding = 1)
        self.d_layer_two = nn.ConvTranspose1d(in_channels = self.hidden_dim, out_channels = self.hidden_dim // 2, kernel_size = 4, stride = 2, padding = 1)
        self.residual_stack = ResidualStack(self.hidden_dim // 2, self.hidden_dim // 2, self.num_residual_layers, self.residual_hidden_dim)
        self.d_layer_three = nn.ConvTranspose1d(in_channels = self.hidden_dim // 2, out_channels = self.hidden_dim // 4, kernel_size = 4, stride = 2, padding = 1)
        self.d_layer_four = nn.ConvTranspose1d(in_channels = self.hidden_dim // 4, out_channels = 1, kernel_size = 4, stride = 2, padding = 1)

        self. decoder_layer = nn.ModuleList([self.d_layer_one, self.d_layer_two ,self.residual_stack, self.d_layer_three])

    def forward(self, x):
        for layer in self.decoder_layer:
            if layer == self.residual_stack:
                x = layer(x)
            else:
                x = layer(x)
                x = relu(x)
        x = self.d_layer_four(x)
        return torch.tanh(x)

#VQ-VAE
class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq, data_variance = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self.data_variance = data_variance

    def forward(self, x):
        z = self.encoder(x)
        z_permute = z.permute(0, 2, 1).contiguous()
        vq_quantize, embedding_idx, vq_loss = self.vq(z_permute)
        vq_quantize_reshape = vq_quantize.permute(0, 2, 1).contiguous()
        output = self.decoder(vq_quantize_reshape)
        if self.data_variance:
            recon_loss = torch.mean(torch.square(output - x)) / self.data_variance
            vq_loss = vq_loss.mean()
            loss = recon_loss + vq_loss
            return {'z': z, 'x': x, 'loss': loss, 'recon_loss': recon_loss, 'vq_loss': vq_loss, 'vq_output': vq_quantize, 'output': output}
        else:
            return {'z': z, 'x': x, 'vq_output': vq_quantize, 'output': output}

#モデルの読み込み
def get_model(data_variance = None):
    encoder = Encoder(in_channels = params.in_channels, hidden_dim = params.hidden_dim, num_residual_layers = params.num_residual_layers, residual_hidden_dim = params.residual_hidden_dim)
    decoder = Decoder(in_channels = params.embedding_dim, hidden_dim = params.hidden_dim, num_residual_layers = params.num_residual_layers, residual_hidden_dim = params.residual_hidden_dim)
    #vq = VectorQuantize(dim = params.embedding_dim, codebook_size = params.num_embeddings)
    vq = ResidualVQ(dim = params.embedding_dim, num_quantizers = 4, codebook_size = params.num_embeddings, kmeans_init = True, kmeans_iters = 10, 
                    in_place_codebook_optimizer = lambda x: torch.optim.Adam(x, lr = params.learning_rate),
                    ema_update = False,
                    learnable_codebook = True)
    model = VQVAE(encoder, decoder, vq, data_variance = data_variance)
    optimizer = torch.optim.Adam(model.parameters(), lr = params.learning_rate)
    return model, optimizer