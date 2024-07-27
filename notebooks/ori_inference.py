import io
import os
import numpy as np

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

import sys

current_directory = os.getcwd()
sys.path.append(current_directory)

import torch
import pandas as pd
import urllib
import tarfile
import whisper
import torchaudio
import librosa
from scipy.io import wavfile
from tqdm import tqdm
from datasets import load_dataset


pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def download(url: str, target_path: str):
    with urllib.request.urlopen(url) as source, open(target_path, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))


# class Fleurs(torch.utils.data.Dataset):
#     """
#     A simple class to wrap Fleurs and subsample a portion of the dataset as needed.
#     """
#     def __init__(self, lang, split="test", subsample_rate=1, device=DEVICE):
#         url = f"https://storage.googleapis.com/xtreme_translations/FLEURS102/{lang}.tar.gz"
#         tar_path = os.path.expanduser(f"/hdd/Gothi_raj/Whisper/data/{lang}.tgz")
#         os.makedirs(os.path.dirname(tar_path), exist_ok=True)

#         if not os.path.exists(tar_path):
#             download(url, tar_path)

#         all_audio = {}
#         with tarfile.open(tar_path, "r:gz") as tar:
#             for member in tar.getmembers():
#                 name = member.name
#                 if name.endswith(f"{split}.tsv"):
#                     labels = pd.read_table(tar.extractfile(member), names=("id", "file_name", "raw_transcription", "transcription", "_", "num_samples", "gender"))

#                 if f"/{split}/" in name and name.endswith(".wav"):
#                     audio_bytes = tar.extractfile(member).read()
#                     all_audio[os.path.basename(name)] = wavfile.read(io.BytesIO(audio_bytes))[1]                    

#         self.labels = labels.to_dict("records")[::subsample_rate]
#         self.all_audio = all_audio
#         self.device = device

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, item):
#         record = self.labels[item]
#         audio = torch.from_numpy(self.all_audio[record["file_name"]].copy())
#         text = record["transcription"]
        
#         return (audio, text)


class Fleurs(torch.utils.data.Dataset):
    """
    A simple class to wrap Fleurs and subsample a portion of the dataset as needed.
    """
    def __init__(self, lang, split="test", subsample_rate=1, device=DEVICE):

        dataset = load_dataset("google/fleurs",f"{lang}_in",split=split,cache_dir='dataset/HF')
        audio_paths = dataset['path']
        ground_truth = dataset['transcription']

        audio_files = []
        for i in range(len(audio_paths)):
            audio_path = audio_paths[i]

            ind=audio_path.rfind('/')
            audio_path = audio_path[:ind]+f'/{split}'+audio_path[ind:]

            audio_files.append(audio_path)

        self.paths = audio_files
        self.transcript = ground_truth
        self.device = device

    def __len__(self):
        return len(self.transcript)

    def __getitem__(self, item):

        y, sr = librosa.load(self.paths[item])
        audio = torch.from_numpy(y)         
        # audio, _ = torchaudio.load(self.paths[item])
        # audio  = 
        # audio = audio.to(self.device)
        # audio = np.array(audio)
        text = self.transcript[item]        
        return (audio, text)

class Kathbath:
    def __init__(self, lang, split="test"):
        self.df = pd.read_csv(lang)
        self.audio_files = self.df['file_path']
        self.ground_truth = self.df['transcript']
        self.paths = self.audio_files
        self.transcripts = self.ground_truth
        self.index = 0  # Initialize an index for iteration

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.paths):
            y, sr = librosa.load(self.paths[self.index])
            audio = torch.from_numpy(y)
            text = self.transcripts[self.index]
            self.index += 1
            return (audio, text)
        else:
            raise StopIteration

def run(lang,language,model,save_path):
    # lang = 'te_in'
    # language = 'telugu'

    assert lang is not None, "Please select a language"
    print(f"Selected language: {language} ({lang})")

    # dataset = Fleurs(lang)  # subsample 10% of the dataset for a quick demo
    dataset = Kathbath(lang)  # subsample 10% of the dataset for a quick demo

    options = dict(language=language, beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)

    references = []
    transcriptions = []

    for audio, text in tqdm(dataset):
        transcription = model.transcribe(audio, **transcribe_options)["text"]
        
        transcriptions.append(transcription)
        references.append(text)

    data = pd.DataFrame(dict(reference=references, transcription=transcriptions))

    data.to_csv(save_path)


model= "medium"

lang = ['te','ml','hi','gu','mr','bn','ta','kn']
languages = ["telugu","malayalam","hindi", "gujarati", "marathi", "bengali", "tamil", "kannada"]

model = whisper.load_model(model)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

for i in range(8):
    save_path = f'Results/openai/medium/kathbath/{lang[i]}.csv'
    path = f'dataset/kathbath/kb_data_clean_wav/{languages[i]}/test/bucket.csv'
    run(path,languages[i],model,save_path)
