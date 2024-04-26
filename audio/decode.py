import torch
import torchaudio
from model import get_model

if __name__ == '__main__':
    wav_path = './sampling/original1.wav'
    model, _ = get_model()
    model.load_state_dict(torch.load('./model/model.pth'))
    model.eval()
    with torch.no_grad():
        wav, sr = torchaudio.load(wav_path)
        wav = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)(wav)
        wav = wav.unsqueeze(0)
        output = model(wav)
        torchaudio.save('./sampling/output.wav', output['output'].squeeze(0), 16000)