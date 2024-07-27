import os
import sys
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import datasets
import torch
import numpy as np
from datasets import Dataset
import wandb

os.environ['WANDB_DISABLED'] = 'true'

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor
)
import pandas as pd
from tqdm import tqdm

import jiwer
from whisper.normalizers import IndicTextNormalizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for model training")
    parser.add_argument("--model_path", type=str, default="trained_model/multi_trail/checkpoint-12500", help="Path to the trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--do_normalize", action='store_true',default = False , help="Whether to normalize the input data")
    parser.add_argument("--language", type=str, default='hi', help="Language of the dataset")
    # parser.add_argument("--data_dir", type=str, default="dataset/kathbath/kb_data_clean_wav/hindi/valid/audio", help="Directory containing the dataset")
    parser.add_argument("--bucket_csv", type=str, default='dataset/kathbath/kb_data_clean_wav/hindi/valid/bucket.csv', help="CSV file containing bucket information")
    # parser.add_argument("--chunk_size", type=int, default=64, help="Size of data chunks")
    parser.add_argument("--save_path", type=str, default='Results/hi_val_medium.csv', help="Path to save the results")
    parser.add_argument("--wer_save_path", type=str, default='Results/wer.txt', help="Path to save the results")
    parser.add_argument("--prompt", type=bool, default=False, help="use prompt?")

    return parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_MODE"]= "disabled"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor ([`WhisperProcessor`])
            The processor used for processing the data.
        decoder_start_token_id (`int`)
            The begin-of-sentence of the decoder.
        forward_attention_mask (`bool`)
            Whether to return attention_mask.
    """

    processor: Any
    decoder_start_token_id: int
    forward_attention_mask: bool
    audio_column_name: str
    do_normalize: bool

    def __call__(
        self, features
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        
        model_input_name = self.processor.model_input_names[0]
        
        features = [
            prepare_dataset(
                feature, 
                audio_column_name=self.audio_column_name, 
                model_input_name=model_input_name,
                feature_extractor=self.processor.feature_extractor,
                do_normalize=self.do_normalize
            ) for feature in features
        ]
        
        input_features = [
            {model_input_name: feature[model_input_name]} for feature in features
        ]

        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        return batch

def prepare_dataset(batch, audio_column_name, model_input_name, feature_extractor, do_normalize):
    # process audio
    sample = batch[audio_column_name]

    # if longer than 30 seconds, truncate.
    # for best score, break long files up
#     if len(sample["array"]) > (16000 * 30):
#         sample["array"] = sample["array"][:16000 * 30]

    inputs = feature_extractor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
        do_normalize=do_normalize,
    )
    # process audio length
    batch[model_input_name] = inputs.get(model_input_name)[0]

    return batch

@dataclass
class Config:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    audio_column_name: str = field(
        default="audio",
        metadata={
            "help": "The name of the dataset column containing the audio data. Defaults to 'audio'"
        },
    )
    num_workers: int = field(
        default=2,
        metadata={
            "help": "The number of workers for preprocessing"
        },
    )
    use_bettertransformer: bool = field(default=False, metadata={
            "help": "Use BetterTransformer (https://huggingface.co/docs/optimum/bettertransformer/overview)"
        })
    do_normalize: bool = field(default=False, metadata={
            "help": "Normalize in the feature extractor"
        })


if __name__ == "__main__":

    CFG = parse_args()

    # class CFG:
    #     model_path = "trained_model/multi_trail/checkpoint-12500"
    #     batch_size = 32
    #     do_normalize = False
    #     language = 'hi'
    #     data_dir = "dataset/kathbath/kb_data_clean_wav/hindi/valid/audio"
    #     bucket_csv = 'dataset/kathbath/kb_data_clean_wav/hindi/valid/bucket.csv'
    #     chunk_size = 64
    #     save_path  = 'Results/hi_val_medium.csv'

    # print(f"Model Path: {CFG.model_path}")
    # print(f"Batch Size: {CFG.batch_size}")
    # print(f"Do Normalize: {CFG.do_normalize}")
    print(f"Language: {CFG.language}")
    # print(f"Data Directory: {CFG.data_dir}")
    # print(f"Bucket CSV: {CFG.bucket_csv}")
    # print(f"Chunk Size: {CFG.chunk_size}")
    # print(f"Save Path: {CFG.save_path}")

    cfg = Config(
        model_name_or_path=CFG.model_path,
        audio_column_name="audio",
        num_workers=2,
        do_normalize=False,
    )

    training_args = Seq2SeqTrainingArguments(
        # Define your training arguments here
        output_dir="./",
        predict_with_generate = True,
        remove_unused_columns=False,
        disable_tqdm=True,
        report_to = None,
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        cfg.model_name_or_path,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        cfg.model_name_or_path,cache_dir = '/hdd/Gothi_raj/HF_model'
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg.model_name_or_path,
        config=config,
        cache_dir = '/hdd/Gothi_raj/HF_model'
    )

    tokenizer = WhisperTokenizer.from_pretrained(cfg.model_name_or_path,cache_dir = '/hdd/Gothi_raj/HF_model',language=CFG.language)

    processor = WhisperProcessor.from_pretrained(cfg.model_name_or_path,cache_dir = '/hdd/Gothi_raj/HF_model')

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    print("language :  ->>>> ",CFG.language)

    model.generation_config.language = CFG.language
    
    # model.generation_config.chunk_length_s = 10

    # model.generation_config.return_timestamps = True

    model.generation_config.no_repeat_ngram_size = 3    
    model.generation_config.num_beams= 3

    #prompting input:
    if CFG.prompt:
        if any(lang in CFG.language for lang in ['hindi', 'gujarati', 'marathi', 'bengali']):
            prompt = "indo"
        elif any(lang in CFG.language for lang in ['tamil', 'telugu', 'kannada', 'malayalam']):
            prompt = 'dra'  
        else:
            print('Error')

        print('Prompting')
        prompt_ids = processor.get_prompt_ids(prompt) if prompt else None
        model.generation_config.prompt_ids = prompt_ids


    df = pd.read_csv(CFG.bucket_csv)

    ds = Dataset.from_dict({"audio": df['file_path']})

    ds = ds.map(lambda x: {"id": Path(x["audio"]).stem, "filesize": os.path.getsize(x["audio"])}, num_proc=cfg.num_workers)

    ds = ds.cast_column(
        cfg.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    # ds = ds.sort("filesize")
    # ds = ds.add_column("idx", range(len(ds)))

    # # save ids
    # ds.remove_columns([x for x in ds.column_names if x != "id"]).to_json("ids.json")

    # mapping = {}
    # for i,row in df.iterrows():
    #     id = row['file_path'].split('/')[-1][:-4]
    #     # print(id)
    #     mapping[id] = row['transcript']

    # ground_truth = []

    # for id in ds['id']:
    #     ground_truth.append(mapping[id])

    ds = ds.add_column('ground_truth',df['transcript'])

    # ds = ds.select(range(32))

    model_input_name = feature_extractor.model_input_names[0]

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=False,
        audio_column_name=cfg.audio_column_name,
        do_normalize=cfg.do_normalize,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=feature_extractor,
        data_collator=data_collator,       
    )


    text_preds = []

    for num, i in enumerate(tqdm(range(0, len(ds), CFG.batch_size), desc="Processing", unit="chunk")):
        ii = min(i+CFG.batch_size, len(ds))
        temp = ds.select(range(i, ii))

        predictions = trainer.predict(temp).predictions
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        text_preds.extend(predictions)
        # pred += predictions
        # print(predictions)
        # break
        # Dataset.from_dict({"idx": temp["idx"]}).to_json(f"vectorized_idxs_{num}.json")
        # np.save(f"preds_{num}.npy", predictions)

    data = pd.DataFrame({'id':ds['id'],'hypothesis':text_preds,'reference':ds['ground_truth']})

    normalizer = IndicTextNormalizer(use_indic_normalizer = True, lang = CFG.language)

    data.to_csv(CFG.save_path)

    data["hypothesis"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference"] = [normalizer(text) for text in data["reference"]]

    data.to_csv(CFG.save_path)

    wer = jiwer.wer(list(data["reference"]), list(data["hypothesis"]))

    print()
    print(f"{CFG.language} WER: {wer * 100:.2f} %")
    with open(CFG.wer_save_path,'a+') as f:
        f.write(f"{CFG.language} WER: {wer * 100:.2f} % \n")
    print()

    data.to_csv(CFG.save_path)