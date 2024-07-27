import os
import sys
import random
import pandas as pd

current_directory = os.getcwd()
sys.path.append(current_directory)

from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from datasets import Audio
import re
from whisper.normalizers import IndicTextNormalizer
from datasets import DatasetDict,Dataset
# /hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/hindi/valid/bucket.csv


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'


def process(language,save_path):

    train_path = f"/hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/{language}/train/bucket.csv"
    dev_path = f"/hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/{language}/valid/bucket.csv"

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

    flg=0
    with open(train_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if flg==0:        
                flg=1
                continue
            line = line.split(',')
            audio.append(line[0])
            transcript.append(line[1])

    train_df = pd.DataFrame({'path':audio,'transcription':transcript})


    audio = []
    transcript= []

    flg=0
    with open(dev_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            # print(line)
            if flg==0:        
                flg=1
                continue
            line = line.split(',')
            audio.append(line[0])
            transcript.append(line[1])

    dev_df = pd.DataFrame({'path':audio,'transcription':transcript})

    train_data_hf = Dataset.from_pandas(train_df)
    dev_data_hf = Dataset.from_pandas(dev_df)

    dataset = DatasetDict({'train':train_data_hf,'validation':dev_data_hf})

    normalizer = IndicTextNormalizer(use_indic_normalizer = True, lang = language)

    dataset = dataset.map(remove_special_characters)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path,cache_dir='/hdd/Gothi_raj/HF_model')

    # tokenizer = WhisperTokenizer.from_pretrained(model_path, language=language, task="transcribe",cache_dir='/hdd/Gothi_raj/HF_model')

    tokenizer = WhisperTokenizer.from_pretrained(tokenizer_path, language=language, task="transcribe",cache_dir='/hdd/Gothi_raj/HF_model')

    input_str = dataset["train"][0]["transcription"]

    # prompt = "IndoAryan"
    # prompt_ids = tokenizer.get_prompt_ids(prompt)

    labels = tokenizer(input_str).input_ids
    
    print('tokenized sentence length',len(labels))
    
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

    # dataset['train'] = dataset['train'].select(range(100))
    # dataset['validation'] = dataset['validation'].select(range(100))

    # dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)

    dataset['train'] = dataset['train'].map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1,cache_file_name=f"/hdd/Gothi_raj/HF_model/cache_{language}_train.txt",keep_in_memory=False)
    dataset['validation'] = dataset['validation'].map(prepare_dataset, remove_columns=dataset.column_names["validation"], num_proc=1,cache_file_name=f"/hdd/Gothi_raj/HF_model/cache_{language}_val.txt",keep_in_memory=False)

    print(dataset)

    dataset.save_to_disk(save_path)


if __name__ == "__main__":

    tokenizer_path = 'tokenizer/all_tokenizer_125'
    languages = [ "hindi","gujarati", "marathi", "bengali", "tamil", "telugu", "kannada", "malayalam"]
    model_path = "openai/whisper-medium"
 
    for language in languages:
        process(language=language,save_path=f"/hdd2/raj/preprocess/{language}_kathbath_medium_Alltokenized_125_dummy")

