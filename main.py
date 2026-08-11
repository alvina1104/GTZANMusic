from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch
import torch.nn as nn
from torchaudio import transforms
import io
import torch.nn.functional as F
import soundfile as sf
import os
import torchaudio
from torch.utils.data import Dataset


class GTZANAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


classes = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'jazz', 'metal', 'pop', 'reggae', 'rock']

device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_to_index  = {n: i for i, n in enumerate(classes)}
index_to_labels = {i: n for i, n in enumerate(classes)}

transform = transforms.MelSpectrogram(sample_rate=22050, n_mels=64)
max_len   = 500

model = GTZANAudio()
model.load_state_dict(torch.load('gtzan_model.pth', map_location=device))
model.to(device)
model.eval()


class GTZANDataSet(Dataset):
    def __init__(self, data_path, transform, max_len):
        self.data_path = data_path
        self.transform = transform
        self.max_len   = max_len
        self.audios    = []

        print("Checking for corrupted files, please wait...")

        for name in os.listdir(data_path):
            name_path = os.path.join(data_path, name)
            if os.path.isdir(name_path):
                for file in os.listdir(name_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(name_path, file)
                        try:
                            torchaudio.load(file_path)
                            self.audios.append((file_path, name))
                        except Exception:
                            print(f"Skipped (corrupted file): {file}")

        print(f"Done! Successfully loaded {len(self.audios)} files.")

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, index):
        file_path, label = self.audios[index]
        waveform, sr = torchaudio.load(file_path)

        if sr != 22050:
            waveform = transforms.Resample(orig_freq=sr, new_freq=22050)(waveform)

        spec = self.transform(waveform).squeeze(0)

        if spec.shape[1] > self.max_len:
            spec = spec[:, :self.max_len]
        if spec.shape[1] < self.max_len:
            spec = F.pad(spec, (0, self.max_len - spec.shape[1]))

        return spec, label_to_index[label]


gtzan_app = FastAPI()

@gtzan_app.post('/predict/')
async def gtzan_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=422, detail="File is empty")

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf.T)


        if sr != 22050:
            wf = transforms.Resample(orig_freq=sr, new_freq=22050)(wf)

        spec = transform(wf).squeeze(0)

        if spec.shape[1] > max_len:
            spec = spec[:, :max_len]
        if spec.shape[1] < max_len:
            spec = F.pad(spec, (0, max_len - spec.shape[1]))

        spec = spec.unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred     = model(spec)
            pred_ind   = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_labels[pred_ind]
            return {'Index': pred_ind, 'Class': pred_class}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(gtzan_app, host="127.0.0.1", port=8001)