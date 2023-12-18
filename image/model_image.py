import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import params_image as params

relu = nn.ReLU()

class Residual(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_hidden):
        super(Residual, self).__init__()
        self.conv_one = nn.Conv2d(in_channels = in_channels, out_channels = num_residual_hidden, kernel_size = 3, stride = 1, padding = 1)
        self.conv_two = nn.Conv2d(in_channels = num_residual_hidden, out_channels = hidden_dim, kernel_size = 1, stride = 1)

    def forward(self, x):
        h = relu(x)
        h = relu(self.conv_one(h))
        h = self.conv_two(h)
        return x + h

class ResidualStack(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hidden):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([Residual(in_channels, hidden_dim, num_residual_hidden) for _ in range(num_residual_layers)])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_residual_layers = num_residual_layers
        self.residual_hidden_dim = residual_hidden_dim

        self.e_layer_one = nn.Conv2d(in_channels = self.in_channels, out_channels = self.hidden_dim // 2, kernel_size = 4, stride = 2, padding = 1)
        self.e_layer_two = nn.Conv2d(in_channels = self.hidden_dim // 2, out_channels = self.hidden_dim, kernel_size = 4, stride = 2, padding = 1)
        self.e_layer_three = nn.Conv2d(in_channels = self.hidden_dim, out_channels = self.hidden_dim, kernel_size = 3, stride = 1, padding = 1)
        self.residual_stack = ResidualStack(self.hidden_dim, self.hidden_dim, self.num_residual_layers, self.residual_hidden_dim)

        self.encoder_layer = nn.ModuleList([self.e_layer_one, self.e_layer_two, self.e_layer_three])

    def forward(self, x):
        for layer in self.encoder_layer:
            if layer != self.e_layer_three:
                x = layer(x)
                x = relu(x)
            else:
                x = layer(x)
        return self.residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_residual_layers = num_residual_layers
        self.residual_hidden_dim = residual_hidden_dim

        self.d_layer_one = nn.Conv2d(in_channels = self.in_channels, out_channels = self.hidden_dim, kernel_size = 3, stride = 1, padding = 1)
        self.residual_stack = ResidualStack(self.hidden_dim, self.hidden_dim, self.num_residual_layers, self.residual_hidden_dim)
        self.d_layer_two = nn.ConvTranspose2d(in_channels = self.hidden_dim, out_channels = self.hidden_dim // 2, kernel_size = 4, stride = 2, padding = 1)
        self.d_layer_three = nn.ConvTranspose2d(in_channels = self.hidden_dim // 2, out_channels = 3, kernel_size = 4, stride = 2, padding = 1)

        self. decoder_layer = nn.ModuleList([self.d_layer_one, self.residual_stack, self.d_layer_two])

    def forward(self, x):
        for layer in self.decoder_layer:
            if layer == self.d_layer_two:
                x = layer(x)
            else:
                x = layer(x)
                x = relu(x)
        x = self.d_layer_three(x)
        return torch.sigmoid(x)

class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embedding = num_embeddings
        self.commitment_cost = commitment_cost
        self.w = nn.Embedding(self.num_embedding, self.embedding_dim)
        self.w.weight.data.uniform_(-1 / self.num_embedding, 1 / self.num_embedding)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.size()
        input_flattened = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(input_flattened ** 2 , dim = 1, keepdim  = True) - 2 * torch.matmul(input_flattened, self.w.weight.t()) + torch.sum(self.w.weight ** 2, dim = 1))
        encoding_indices = torch.argmax(-distances, 1).unsqueeze(1)
        encoding = torch.zeros(encoding_indices.shape[0], self.num_embedding, device = inputs.device)
        encoding.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encoding, self.w.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        avg_probs = torch.mean(encoding, dim = 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return {'distances': distances, 'quantize': quantized, 'loss': loss,'encoding': encoding, 'encoding_indices': encoding_indices, 'perplexity': perplexity}

class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq, pre_vq_conv1, data_variance, name = None):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self.pre_vq_conv1 = pre_vq_conv1
        self.data_variance = data_variance

    def forward(self, x):
        z = self.pre_vq_conv1(self.encoder(x))
        vq_output = self.vq(z)
        output = self.decoder(vq_output['quantize'])
        reconstructed_error = torch.mean(torch.square(output - x)) / self.data_variance
        loss = reconstructed_error + vq_output['loss']
        return {'z': z, 'x': x, 'loss': loss, 'reconstructed_error': reconstructed_error, 'vq_output': vq_output, 'output': output}

def get_model(data_variance):
    encoder = Encoder(in_channels = params.in_channels, hidden_dim = params.hidden_dim, num_residual_layers = params.num_residual_layers, residual_hidden_dim = params.residual_hidden_dim)
    decoder = Decoder(in_channels = params.embedding_dim, hidden_dim = params.hidden_dim, num_residual_layers = params.num_residual_layers, residual_hidden_dim = params.residual_hidden_dim)
    pre_vq_conv1 = nn.Conv2d(in_channels = params.hidden_dim, out_channels = params.embedding_dim, kernel_size = 1, stride = 1)
    vq = VectorQuantizer(embedding_dim = params.embedding_dim, num_embeddings = params.num_embeddings, commitment_cost = params.commitment_cost)

    model = VQVAE(encoder, decoder, vq, pre_vq_conv1, data_variance = data_variance)
    optimizer = torch.optim.Adam(model.parameters(), lr = params.lr)
    return model, optimizer