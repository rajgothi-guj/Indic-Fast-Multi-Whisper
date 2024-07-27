import os
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm

from whisper.decoding import DecodingTask,DecodingOptions,DecodingResult

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from torch import Tensor
if TYPE_CHECKING:
    from .model import Whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass(frozen=True)
class FeatureResult:
    audio_features: Tensor
    Language: str= 'en'

class audioDataset(torch.utils.data.Dataset):
    """
    A custom dataset class that takes a list of audio paths and returns the mel spectogram.
    Note:  It will trim/pad the audio to 30 seconds, as per whisper architecture.
    """

    def __init__(self, audio_paths, device=DEVICE):
        self.audio_paths = audio_paths
        self.device = device

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, item):
        audio_path = self.audio_paths[item]
        audio, sample_rate = torchaudio.load(audio_path)
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        return mel
    
@torch.no_grad()
def extractFeature(
    model: "Whisper",
    mel: Tensor,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    
    if single := mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)

    decodingObject = DecodingTask(model, options)
    audio_features: Tensor = decodingObject._get_audio_features(mel)
    audio_features = audio_features[:: decodingObject.n_group]
    
    fields = (audio_features)

    if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

    result = [
            FeatureResult(audio_features=features)
            for features in fields
        ]

    return result[0] if single else result

def get_audio_files(file_path_or_folder):
    audio_files = []
    if os.path.isdir(file_path_or_folder):
        for file_name in os.listdir(file_path_or_folder):
            file_path = os.path.join(file_path_or_folder, file_name)
            if os.path.isfile(file_path) and file_name.endswith(('.wav', '.mp3')):
                audio_files.append(file_path)
    elif os.path.isfile(file_path_or_folder) and file_path_or_folder.endswith(('.wav', '.mp3')):
        audio_files.append(file_path_or_folder)
    else:
        raise ValueError("Invalid file path or folder path provided.")
    return audio_files

def main():

    model_name = "base.en"

    path = 'D:\Code\Whisper\whisper\Audios'
    # path = 'D:\Code\Whisper\whisper\Audios\\aa_3-b-2_kv_bhandupshift1_01092022-000000-1_EN-OL-RC-234_1.wav'

    audio_paths = get_audio_files(path) #path can be folder or audio path...

    model = whisper.load_model(model_name)

    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    
    dataset = audioDataset(audio_paths=audio_paths)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    audio_features = []

    for mels in tqdm(loader):
        # audio_features= DecodingTask._get_audio_features(mel)  # encoder forward pass
        results = extractFeature(model, mels, options)
        audio_features.extend([result.audio_features for result in results])  # we need because whisper only take 30 seconds..

    print("Shape of features: ",audio_features[0].shape)
    print(f"Successfully extracted Features of {len(audio_paths)} audio(s).")

if __name__ == "__main__":
    main()
