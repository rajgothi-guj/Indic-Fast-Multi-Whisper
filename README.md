# Indic-Fast-Multi-Whisper

This repo contains the code of Indic-Fast-Multi-Whisper, written by Raj Gothi.

## Folder Structure
```
Whisper/
│   ├── Config/  # contains the yaml file for train and test.
│   ├── dataset/ # stored the downloaded dataset of fleurs and kathbath
│   ├── features/ # to get audiofeatures from whisper encoder (done initially, not required for ASR)
│   ├── finetuning/ # It contains all the code of proposed technique with fine-tuning script.
│   ├── IndicSUPERB/ # cloned repo of kathbath dataset
│   ├── notebooks/ # This is jupyter notebook given by Original whisper code...
│   ├── Results/  # It stores the results for each of the inferences... model name / dataset / csv,wer,log files...
│   ├── tokenizer/ # Stored new tokenizers
│   ├── trained_model/ # It stores the fine-tuned models
│   ├── transformers/ # transformers library code
│   ├── whisper/ # original github whisper code
```



Use Whisper Enviroment for finetuning :  /hdd/Gothi_raj/envs/whisper

Use wt Enviroment for Inference: /hdd/Gothi_raj/envs/wt




## Dataset Download

dataset folder contain the downloaded dataset.
```
datasets/
│   ├── HF/
│   ├── Kathbath/     
```

### Fleurs

Dataset link : https://huggingface.co/datasets/google/fleurs

```python
python fleurs_download.py
```

### Kathbath

Dataset Link: https://github.com/AI4Bharat/IndicSUPERB

Follow this instructions https://github.com/AI4Bharat/IndicSUPERB#data-preprocessing after downloading dataset.

It will create bucket.csv file for each language.

```
datasets/
│   ├── HF/
│   ├── Kathbath/
|        ├──  kb_data_clean_wav/
|           ├── hindi
|                ├── train 
|                    ├── bucket.csv
|                ....
```

## Dataset Pre-processing

It extract the log-mel features from the audio and encode the ground truth text into numbers, which is required for fine-tuning.

### kathbath

```python
python finetuning/kathbath.py
```

set model_path based on the model name (as in Huggingface)...

model_path = "openai/whisper-medium" # openai/whisper-large-v3


and save_path argument to process function...

It will store the pre-processed dataset to mentioned save_path location for all the languages,which will be used in fine-tuning step.

Code will read the bucket.csv files of all the language that we created in earlier step.

make sure to remove cached dataset at location "/hdd/Gothi_raj/HF_model/"   (Enhancement: Now code will remove cache file automatically.)

ex. rm cache_gujarati_train.txt cache_gujarati_val.txt



### Fleurs

```python
python finetuning/fleurs.py
```

set model_path based on the model name (as in Huggingface)...

model_path = "openai/whisper-medium" # openai/whisper-large-v3


and save_path argument to process function...

It will store the pre-processed dataset to mentioned save_path location for all the languages,which will be used in fine-tuning step.

Code will read the dowloaded fleurs dataset of all the language that we downloaded in earlier step.

make sure to remove cached dataset at location "/hdd/Gothi_raj/HF_model/"  (Enhancement: Now code will remove cache file automatically.)

ex. rm cache_gu_train.txt cache_gu_val.txt
(Here it is language code as per fleurs.. Look at code..)


## Fine-tuning

```
Config/
│   ├── HF_train/
|        ├──  multi.yaml
         ├──  multi_token.yaml
         ├──  multi_lora.yaml
    ├── test
        ....
```

This config folder contains the yaml file which is useful to set parameters for training the model.

### FT

set parameters in multi.yaml file, change dataset path, output-dir path (ex. trained_model/fleurs_medium_FT), prompting=False

```python
torchrun --nproc_per_node=4 finetuning/hf_finetune.py
```


### Prompt FT

set parameters in multi.yaml file, change dataset path, output-dir path (ex. trained_model/fleurs_medium_Prompt) and prompting=True

It will add prompting based on the language name pass in multi.yaml file...

```python
torchrun --nproc_per_node=4 finetuning/hf_finetune.py
```


### Tokenizer FT

BPE blog : https://huggingface.co/learn/nlp-course/en/chapter6/5

set this 3 variables inside the ```finetuning/create_tokenizer.py```

```bash
tokenizer_save_path = "tokenizer/all_tokenizer_250"
model_path = "openai/whisper-medium"
vocab_size = 250
```

Run to create tokenizer first which will be stored to tokenizer_save_path...

```python
python finetuning/create_tokenizer.py
```

data-preprocessing step to tokenized ground truth setences...

```python
python finetuning/preprocessTokenizerDataset.py
```

set model_path based on the model name (as in Huggingface)...

model_path = "openai/whisper-medium" # openai/whisper-large-v3

and save_path argument to process function...

It will store the pre-processed dataset to mentioned save_path location for all the languages,which will be used in tokenizer fine-tuning step.

Code will read the bucket.csv files of all the language that we created in earlier step.

make sure to remove cached dataset at location "/hdd/Gothi_raj/HF_model/"  (Enhancement: Now code will remove cache file automatically.)

ex. rm cache_gujarati_train.txt cache_gujarati_val.txt


fine-tuning of Tokenizer FT model

set all the parameters in multi_token.yaml file...

```python
torchrun --nproc_per_node=3 finetuning/hf_finetune_newtoken.py
```

### Lora FT

set parameters in multi_lora.yaml file, change dataset path, output-dir path ...

It will add prompting based on the language name pass in multi.yaml file...

```python
python finetuning/hf_finetune_lora.py
```


## Inference


```
Config/
│   ├── HF_train/
|        ├──  ....
    ├── test
         ├──  run.py #kathbath testing
         ├──  fleurs_run.py
         ├──  spni_run.py
```

set the required parameters in each testing run file... All the parameters are self-explanatory...

Kathbath evaluation:

```python
python Config/test/run.py
```

Fleurs evaluation:

```python
python Config/test/fleurs_run.py
```

SPNI evaluation:

SPNI_bucket.ipynb contains the code for how to create bucket.csv file.

```python
python Config/test/spni_run.py
```

if you don't want to use beam_search and prompting then make sure you comment below line...

    # "--prompt",str(prompt),
    # "--beam_search",str(beam_search)

All the experiment is done with batch_size=32 and beam_search = 5 (medium), 3 (large)

if you want to measure time then comment beam search line as above and set batch size to 1. 

