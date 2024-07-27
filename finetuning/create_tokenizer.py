#Data Pre-processing part
import os
import random
import pandas as pd
from datasets import DatasetDict,Dataset
# new tokenizer
from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer)
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer



def run(train_path,text_save_path):
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

    train_data_hf = Dataset.from_pandas(train_df)
    dataset = DatasetDict({'train':train_data_hf})

    train_transcription = dataset['train']['transcription']
    train_transcription = set(train_transcription)

    with open(text_save_path,'w') as f:
        for sentence in train_transcription:
            f.write(sentence+"\n")

    new_tokenizer = Tokenizer(models.BPE())
    new_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=250)
    train_file = text_save_path
    new_tokenizer.train([train_file], trainer=trainer)
    new_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    new_tokenizer.decoder = decoders.ByteLevel()

    # new_tokenizer = GPT2TokenizerFast(tokenizer_object=new_tokenizer)
    # new_tokenizer.save_pretrained("new_tokenizer_gpt2")
    # new_tokenizer

    # gpt2 tokenizer
    # gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # print(len(gpt2_tokenizer.get_vocab()))
    # gpt2_tokenizer

    # merge the vocabulary for the extended tokenizer
    vocab_tokens = list(new_tokenizer.get_vocab())
    decoded_tokens = [new_tokenizer.decoder.decode([token]) for token in vocab_tokens]
    return decoded_tokens
    # print(len(vocab_tokens), len(decoded_tokens))
    # gpt2_tokenizer.add_tokens(decoded_tokens)

    # gpt2_tokenizer.save_pretrained(tokenizer_save_path)
    # print(len(gpt2_tokenizer.get_vocab()))


def create_tokenizer(decoded_tokens,model_path,tokenizer_save_path):

    tokenizer = WhisperTokenizer.from_pretrained(model_path, task="transcribe")

    tokenizer.add_tokens(decoded_tokens)

    print(len(tokenizer.get_vocab()))

    tokenizer.save_pretrained(tokenizer_save_path)

    # input_str = dataset["train"][0]["transcription"]

    # # prompt = "IndoAryan"
    # # prompt_ids = tokenizer.get_prompt_ids(prompt)

    # labels = tokenizer(input_str).input_ids

    # print('tokenized sentence length',len(labels))
    # # tokenizer = WhisperTokenizer.from_pretrained(model_path, language="Bengali", task="transcribe")
    # # tokenizer.add_tokens(decoded_tokens)
    # # labels = [51, 71, 72, 82, 53552, 82, 53869, 340, 2455, 83, 50257]

    # decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    # decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

    # print(f"Input:                 {input_str}")
    # print(f"Decoded w/ special:    {decoded_with_special}")
    # print(f"Decoded w/out special: {decoded_str}")
    # print(f"Are equal:             {input_str == decoded_str}")

# train_path = ["/hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/gujarati/train/bucket.csv",
#                 "/hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/gujarati/train/bucket.csv",
#                 ]
# text_save_path = ["dataset/kathbath/kb_data_clean_wav/gujarati/all_text.txt",
#                 ]

lang = ["hindi", "gujarati", "marathi", "bengali", "tamil", "telugu", "kannada", "malayalam"]
tokenizer_save_path = "tokenizer/all_tokenizer_125"
model_path = "openai/whisper-medium"
# language = 'gujarati'

decoded_tokens = []

for i in range(len(lang)):    
    path = f'/hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav/{lang[i]}/train/bucket.csv'
    text_save_path = f"dataset/kathbath/kb_data_clean_wav/{lang[i]}/all_text.txt"
    decoded_tokens += run(path,text_save_path)

create_tokenizer(decoded_tokens,model_path,tokenizer_save_path)
print('Done')
