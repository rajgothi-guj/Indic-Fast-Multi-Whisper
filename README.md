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


## Fine-tuning


### FT


### Prompt FT


### Tokenizer FT


## Inference


