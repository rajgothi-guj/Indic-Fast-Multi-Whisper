import os
import sys
import random
import pandas as pd
from datasets import load_dataset

current_directory = os.getcwd()
sys.path.append(current_directory)

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import Audio
import re
from whisper.normalizers import IndicTextNormalizer
from datasets import DatasetDict,Dataset,concatenate_datasets


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


def process(language,fleurs,save_path):

    def remove_special_characters(batch):
        batch["transcription"] = normalizer(batch["transcription"])
        return batch

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["path"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    audio = []
    transcript= []

    dataset = load_dataset("google/fleurs",fleurs,cache_dir='/hdd/Gothi_raj/dataset/dataset/HF')
    audio_paths = dataset['train']['path']
    transcript = dataset['train']['transcription']
    for i in range(len(audio_paths)):
        audio_path = audio_paths[i]
        audio_path = audio_path.replace('Whisper/dataset','dataset/dataset')
        ind=audio_path.rfind('/')
        audio_path = audio_path[:ind]+f'/train'+audio_path[ind:]
        audio.append(audio_path)

    train_df = pd.DataFrame({'path':audio,'transcription':transcript})

    audio = []
    transcript = []

    audio_paths = dataset['validation']['path']
    transcript = dataset['validation']['transcription']
    for i in range(len(audio_paths)):
        audio_path = audio_paths[i]
        # audio_path = audio_path.replace('Whisper/dataset','dataset/dataset')
        ind=audio_path.rfind('/')
        audio_path = audio_path[:ind]+f'/dev'+audio_path[ind:]
        audio.append(audio_path)

    dev_df = pd.DataFrame({'path':audio,'transcription':transcript})

    print(language + "started")
    train_data_hf = Dataset.from_pandas(train_df)
    dev_data_hf = Dataset.from_pandas(dev_df)

    dataset = DatasetDict({'train':train_data_hf,'validation':dev_data_hf})

    normalizer = IndicTextNormalizer(use_indic_normalizer = True, lang = language)

    dataset = dataset.map(remove_special_characters)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path,cache_dir='/hdd/Gothi_raj/HF_model')

    tokenizer = WhisperTokenizer.from_pretrained(model_path, language=language, task="transcribe",cache_dir='/hdd/Gothi_raj/HF_model')

    input_str = dataset["train"][0]["transcription"]

    # prompt = "IndoAryan"
    # prompt_ids = tokenizer.get_prompt_ids(prompt)

    labels = tokenizer(input_str).input_ids

    # tokenizer = WhisperTokenizer.from_pretrained(model_path, language="Bengali", task="transcribe")
    # tokenizer.add_tokens(decoded_tokens)
    # labels = [51, 71, 72, 82, 53552, 82, 53869, 340, 2455, 83, 50257]

    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

    print(f"Input:                 {input_str}")
    print(f"Decoded w/ special:    {decoded_with_special}")
    print(f"Decoded w/out special: {decoded_str}")
    print(f"Are equal:             {input_str == decoded_str}")

    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    # print(dataset)

    dataset['train'] = dataset['train'].map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1,cache_file_name=f"/hdd/Gothi_raj/HF_model/cache_{language}_train.txt",keep_in_memory=False)
    dataset['validation'] = dataset['validation'].map(prepare_dataset, remove_columns=dataset.column_names["validation"], num_proc=1,cache_file_name=f"/hdd/Gothi_raj/HF_model/cache_{language}_val.txt",keep_in_memory=False)
    
    print(dataset)

    dataset.save_to_disk(save_path)

if __name__ == "__main__":
    languages = [ "hi","gu", "mr", "bn", "ta", "te", "kn", "ml"]
    model_path = "openai/whisper-medium"
    for language in languages:
        process(language=language,fleurs = language+"_in",save_path=f"/hdd2/raj/preprocess/{language}_fleurs_medium")
        