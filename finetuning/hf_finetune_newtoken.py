import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets, load_from_disk
import yaml
import argparse
import wandb
import os
import sys
import numpy as np
import random

current_directory = os.getcwd()
sys.path.append(current_directory)
from whisper.normalizers import IndicTextNormalizer

# from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

# /hdd2/raj/preprocess

# conda activate /hdd/Gothi_raj/envs/wt
# /hdd2/kumud/envs/spk/bin/python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url 
# https://download.pytorch.org/whl/cu117/

ngpu=4  # number of GPUs to perform distributed training on.

os.environ["WANDB_PROJECT"] = "multilingual"
os.environ['WANDB_DISABLED'] = 'true'

# %env WANDB_ENTITY=
# %env WANDB_PROJECT=your-project-name

# torchrun --nproc_per_node=2 Whisper/finetuning/hf_finetune.py \
# --model_name vasista22/whisper-hindi-base \
# --language Hindi \
# --sampling_rate 16000 \
# --num_proc 2 \
# --train_strategy steps \
# --learning_rate 3e-3 \
# --warmup 1000 \
# --train_batchsize 16 \
# --eval_batchsize 8 \
# --num_steps 10000 \
# --resume_from_ckpt None \
# --output_dir op_dir_steps \
# --train_datasets mozilla-foundation/common_voice_11_0 mozilla-foundation/common_voice_11_0 \
# --train_dataset_configs hi hi \
# --train_dataset_splits train validation \
# --train_dataset_text_columns sentence sentence \
# --eval_datasets "google/fleurs" \
# --eval_dataset_configs hi_in \
# --eval_dataset_splits test \
# --eval_dataset_text_columns transcription

# torchrun --nproc_per_node=2 Whisper/finetuning/hf_finetune.py --model_name openai/whisper-medium --language Hindi --sampling_rate 16000 --num_proc 2 --train_strategy steps --learning_rate 3e-3 --warmup 1000 --train_batchsize 16 --eval_batchsize 8 --num_steps 10000 --resume_from_ckpt None --output_dir op_dir_steps --train_datasets mozilla-foundation/common_voice_11_0 mozilla-foundation/common_voice_11_0 --train_dataset_configs hi hi --train_dataset_splits train validation --train_dataset_text_columns sentence sentence --eval_datasets "google/fleurs" --eval_dataset_configs hi_in --eval_dataset_splits test --eval_dataset_text_columns transcription

# torchrun --nproc_per_node=3 finetuning/hf_finetune_newtoken.py

# https://github.com/huggingface/distil-whisper/blame/914dcdf3919552d5a3826a9d5db99b059ddcc16e/training/run_distillation.py#LL389

# whisper prompting: https://github.com/huggingface/transformers/issues/24272
# prompt-whisper HF code: https://github.com/kehanlu/Prompt-Whisper/blob/main/prompt_whisper.py

# tokenizer.set_prefix_tokens(language=prefix_language, task=prefix_task, predict_timestamps=prefix_timestamps)
# https://github.com/NbAiLab/nb-whisper/blob/352bf2d0efb073405c90fb4ef048a5d52b6128b6/run_nb_flax_speech_recognition_seq2seq_streaming_dev.py#L579-L582
#######################     ARGUMENT PARSING        #########################


def load_config(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

def print_chosen_variables(args):
    for key, value in vars(args).items():
        print(f"{key}: {value}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# if __name__ == "__main__":
args = load_config("Config/HF_train/multi_token.yaml")
# print_chosen_variables(args)


# wandb.init(
#     entity = 'rajsony',
#     project = 'multilingual',
#     name = f'{args.output_dir}',
#     config = args
# )

set_seed(args.seed)
# args = parser.parse_args()


if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps and epoch.')

if len(args.train_datasets) == 0:
    raise ValueError('No train dataset has been passed')
if len(args.eval_datasets) == 0:
    raise ValueError('No evaluation dataset has been passed')

# if len(args.train_datasets) != len(args.train_dataset_configs):
#     raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_configs. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_configs)} for train_dataset_configs.")
# if len(args.eval_datasets) != len(args.eval_dataset_configs):
#     raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_configs. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_configs)} for eval_dataset_configs.")

# if len(args.train_datasets) != len(args.train_dataset_splits):
#     raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_splits. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_splits)} for train_dataset_splits.")
# if len(args.eval_datasets) != len(args.eval_dataset_splits):
#     raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_splits. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_splits)} for eval_dataset_splits.")

# if len(args.train_datasets) != len(args.train_dataset_text_columns):
#     raise ValueError(f"Ensure that the number of entries in the list of train_datasets equals the number of entries in the list of train_dataset_text_columns. Received {len(args.train_datasets)} entries for train_datasets and {len(args.train_dataset_text_columns)} for train_dataset_text_columns.")
# if len(args.eval_datasets) != len(args.eval_dataset_text_columns):
    # raise ValueError(f"Ensure that the number of entries in the list of eval_datasets equals the number of entries in the list of eval_dataset_text_columns. Received {len(args.eval_datasets)} entries for eval_datasets and {len(args.eval_dataset_text_columns)} for eval_dataset_text_columns.")

print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
print('ARGUMENTS OF INTEREST:')
print(vars(args))
print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
# normalizer = BasicTextNormalizer()
normalizer = IndicTextNormalizer(use_indic_normalizer = False)

#############################       MODEL LOADING       #####################################

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name,cache_dir='/hdd/Gothi_raj/HF_model')
tokenizer = WhisperTokenizer.from_pretrained(args.tokenizer_name, task="transcribe",cache_dir='/hdd/Gothi_raj/HF_model')

processor = WhisperProcessor(tokenizer=tokenizer,feature_extractor=feature_extractor)
processor.save_pretrained(args.output_dir)

# processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe",cache_dir='/hdd/Gothi_raj/HF_model')

model = WhisperForConditionalGeneration.from_pretrained(args.model_name,cache_dir='/hdd/Gothi_raj/HF_model')

print("Size of the vocab: ",len(tokenizer.get_vocab()))

# add new random embeddings for the appended tokens
model.resize_token_embeddings(len(tokenizer))
print('Resized the embeddings')

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that config.decoder_start_token_id is correctly defined")

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
# model.config.apply_spec_augment = True
# model.config.mask_time_prob = 0.05
# model.config.mask_feature_prob = 0.05



if gradient_checkpointing:
    model.config.use_cache = False

# indo_prompt_ids = tokenizer.get_prompt_ids('indo')
# dra_prompt_ids = tokenizer.get_prompt_ids('dra')


def get_prompt_ids(text: str, return_tensors="np"):
    """Converts prompt text to IDs that can be passed to [`~WhisperForConditionalGeneration.generate`]."""
    batch_encoding = tokenizer("<|startofprev|>", " " + text.strip(), add_special_tokens=False)

    # Check for special tokens
    # prompt_text_ids = batch_encoding["input_ids"][1:]
    # special_token_id = next((x for x in prompt_text_ids if x >= tokenizer.all_special_ids[0]), None)
    # if special_token_id is not None:
    #     token = tokenizer.convert_ids_to_tokens(special_token_id)
    #     raise ValueError(f"Encountered text in the prompt corresponding to disallowed special token: {token}.")

    batch_encoding.convert_to_tensors(tensor_type=return_tensors)
    return batch_encoding["input_ids"]

############################        DATASET LOADING AND PREP        ##########################

def load_all_datasets(split):    
    combined_dataset = []
    if split == 'train':
        # for i, ds in enumerate(args.train_datasets):
        #     dataset = load_dataset(ds, args.train_dataset_configs[i], split=args.train_dataset_splits[i],cache_dir='/hdd/Gothi_raj/Whisper/dataset/HF')
        #     dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
        #     if args.train_dataset_text_columns[i] != "sentence":
        #         dataset = dataset.rename_column(args.train_dataset_text_columns[i], "sentence")
        #     dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
        #     combined_dataset.append(dataset)
        for i, ds in enumerate(args.train_datasets):
            dataset = load_from_disk(ds)
            dataset = dataset[args.train_dataset_splits[i]]

            if args.prompting:
                if any(lang in ds for lang in ['hindi', 'gujarati', 'marathi', 'bengali']):
                    prompt = "indo"
                elif any(lang in ds for lang in ['tamil', 'telugu', 'kannada', 'malayalam']):
                    prompt = 'dra'
                else:
                    print("Error")

                # prompt_ids = tokenizer.get_prompt_ids(prompt)

                prompt_column = [prompt] * len(dataset)
                dataset = dataset.add_column("prompt", prompt_column)

            combined_dataset.append(dataset)

    elif split == 'eval':
        # for i, ds in enumerate(args.eval_datasets):
        #     dataset = load_dataset(ds, args.eval_dataset_configs[i], split=args.eval_dataset_splits[i],cache_dir='/hdd/Gothi_raj/Whisper/dataset/HF')
        #     dataset = dataset.cast_column("audio", Audio(args.sampling_rate))
        #     if args.eval_dataset_text_columns[i] != "sentence":
        #         dataset = dataset.rename_column(args.eval_dataset_text_columns[i], "sentence")
        #     dataset = dataset.remove_columns(set(dataset.features.keys()) - set(["audio", "sentence"]))
            # combined_dataset.append(dataset)
        for i, ds in enumerate(args.eval_datasets):
            dataset = load_from_disk(ds)
            dataset = dataset[args.eval_dataset_splits[i]]
            dataset = dataset.shuffle(seed=args.seed)
            dataset = dataset.select(range(750))

            if args.prompting:
                if any(lang in ds for lang in ['hindi', 'gujarati', 'marathi', 'bengali']):
                    prompt = "indo"
                elif any(lang in ds for lang in ['tamil', 'telugu', 'kannada', 'malayalam']):
                    prompt = "dra"
                else:
                    print("Error")

                # prompt_ids = tokenizer.get_prompt_ids(prompt)

                prompt_column = [prompt] * len(dataset)
                dataset = dataset.add_column("prompt", prompt_column)

            combined_dataset.append(dataset)
        
    ds_to_return = concatenate_datasets(combined_dataset)
    ds_to_return = ds_to_return.shuffle(seed=args.seed)
    return ds_to_return

def prepare_dataset(batch):
    # load and (possibly) resample audio data to 16kHz
    # audio = batch["audio"]

    # # compute log-Mel input features from input audio array 
    # batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # # compute input length of audio sample in seconds
    # batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # # optional pre-processing steps
    # transcription = batch["sentence"]
    # if do_lower_case:
    #     transcription = transcription.lower()
    # if do_remove_punctuation:
    #     transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    # batch["labels"] = processor.tokenizer(transcription).input_ids
    # if batch['prompt'] == "indo":
    #     prompt_ids  = indo_prompt_ids
    # else:
    #     prompt_ids  = dra_prompt_ids

    # prompt_ids = tokenizer.get_prompt_ids(batch['prompt'])
    prompt_ids = get_prompt_ids(batch['prompt'])
    new_list = []
    new_list.extend(prompt_ids)
    new_list.extend(batch["labels"])
    batch["labels"] = new_list

    # print(tokenizer.decode(new_list, skip_special_tokens=False))

    # batch["labels"] = prompt_ids + batch["labels"]
    return batch

max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(labels):
    # return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length
    return 0 < len(labels) < max_label_length


print('DATASET PREPARATION IN PROGRESS...')
raw_dataset = DatasetDict()
raw_dataset["train"] = load_all_datasets('train')
raw_dataset["eval"] = load_all_datasets('eval')

if args.prompting:
    raw_dataset = raw_dataset.map(prepare_dataset, num_proc=args.num_proc)

raw_dataset = raw_dataset.filter(
    is_in_length_range,
    input_columns=["labels"],
    num_proc=args.num_proc,
) 

###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int = model.config.decoder_start_token_id

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

     # shift labels to the right to get decoder input ids
        labels = labels_batch["input_ids"]
        decoder_input_ids = labels[:, :-1]
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        # # replace padding with -100 to ignore loss correctly
        # labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # # if bos token is appended in previous tokenization step,
        # # cut bos token here as it's append later anyways
        # if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
        #     labels = labels[:, 1:]

        # batch["labels"] = labels

        # print(tokenizer.decode(labels[0], skip_special_tokens=False))

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print('DATASET PREPARATION COMPLETED')


metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id


    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # print(label_str)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


###############################     TRAINING ARGS AND TRAINING      ############################

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=5,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        report_to=["wandb"],
        # load_best_model_at_end=True,
        # metric_for_best_model="wer",
        # greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="steps",
        num_train_epochs = args.num_epochs,
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        # max_steps=args.num_steps,
        save_total_limit=3,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=440,
        logging_steps=200,
        # report_to=["wandb"],
        load_best_model_at_end=True,
        # metric_for_best_model="wer",
        # greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
    )

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["eval"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

print('TRAINING IN PROGRESS...')
trainer.train()

trainer.save_model(args.output_dir)

print('DONE TRAINING')