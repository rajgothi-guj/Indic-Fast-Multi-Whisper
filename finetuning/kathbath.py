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
from datasets import DatasetDict,Dataset,concatenate_datasets
# /hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/hindi/valid/bucket.csv

train_path = "/hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/hindi/train/bucket.csv"
dev_path = "/hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/hindi/valid/bucket.csv"
language = 'hindi'
model_path = "openai/whisper-large-v3"
# model_path = "openai/whisper-medium"
# save_path = '/hdd2/raj/preprocess/hindi_kathbath_largev3'
save_path = 'HFDataset/hindi_kathbath_largev3'

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

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


if __name__ == "__main__":
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

    # dataset['train'] = dataset['train'].select(range(10000))
    # dataset['validation'] = dataset['validation'].select(range(1000))
    
    # dataset = dataset.flatten_indices()

    # Function to split dataset into chunks
    # def split_dataset(dataset, chunk_size):
    #     return [dataset.shard(num_shards=len(dataset)//chunk_size, index=i) for i in range((len(dataset) + chunk_size - 1) // chunk_size)]

    # # Function to process each chunk
    # def process_chunk(chunk):
    #     return chunk.map(prepare_dataset, remove_columns=chunk.column_names, num_proc=1)

    # # Split the dataset into chunks
    # chunk_size = 25000
    # chunks = split_dataset(dataset['train'], chunk_size)

    # # Process each chunk separately
    # processed_chunks = [process_chunk(chunk) for chunk in chunks]

    # # Concatenate the processed chunks back together
    # processed_dataset = concatenate_datasets(processed_chunks)

    #  dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1)
    # dataset['train']  = processed_dataset
    # dataset['validation'] = dataset['validation'].map(prepare_dataset, remove_columns=dataset.column_names["validation"], num_proc=1)
    
    # dataset = dataset.shard(num_shards=4, index=0)
    
    # dataset['train']
    
    # dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1,cache_file_name="/hdd/Gothi_raj/HF_model",keep_in_memory=False)
    print(dataset)

    dataset['train'] = dataset['train'].map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=1,cache_file_name=f"/hdd/Gothi_raj/HF_model/cache_{language}_train.txt",keep_in_memory=False)
    dataset['validation'] = dataset['validation'].map(prepare_dataset, remove_columns=dataset.column_names["validation"], num_proc=1,cache_file_name=f"/hdd/Gothi_raj/HF_model/cache_{language}_val.txt",keep_in_memory=False)
    
    print(dataset)

    dataset.save_to_disk(save_path)

