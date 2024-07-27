# Indic-Fast-Multi-Whisper

This repo contains the code of Indic-Fast-Multi-Whisper, written by Raj Gothi.

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

### kathbath

```python
python finetuning/kathbath.py
```
    model_path = "openai/whisper-medium" # openai/whisper-large-v3

set model_path based on the model name (as in Huggingface)...

and save_path argument to process function...

It will store the pre-processed dataset to mentioned save_path location for all the languages,which will be used in fine-tuning step.

Code will read the bucket.csv files of all the language that we created in earlier step.

make sure to remove cached dataset at location /hdd/Gothi_raj/HF_model/

ex. rm cache_gujarati_train.txt cache_gujarati_val.txt


### Fleurs

```python
python finetuning/fleurs.py
```

    model_path = "openai/whisper-medium" # openai/whisper-large-v3

set model_path based on the model name (as in Huggingface)...

and save_path argument to process function...

It will store the pre-processed dataset to mentioned save_path location for all the languages,which will be used in fine-tuning step.

Code will read the dowloaded fleurs dataset of all the language that we downloaded in earlier step.

make sure to remove cached dataset at location /hdd/Gothi_raj/HF_model/

ex. rm cache_gu_train.txt cache_gu_val.txt
(Here it is language code as per fleurs.. Look at code..)


## Fine-tuning


### FT


### Prompt FT


### Tokenizer FT


## Inference


