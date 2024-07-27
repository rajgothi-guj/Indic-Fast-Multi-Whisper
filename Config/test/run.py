import os
import subprocess
import time

# List of language IDs
languages = [ "hindi","gujarati", "marathi", "bengali", "tamil", "telugu", "kannada", "malayalam"]
# languages = ["tamil", "telugu", "kannada", "malayalam"]
# languages = [ "gujarati"]

# Common parameters
model_path = "trained_model/Alltokenized_Medium_125"
# model_path = "trained_model/gu_token"
batch_size = 1
beam_search = 1
# do_normalize = False

# Base directories and file names
data_dir_base = "/hdd/Gothi_raj/Whisper/dataset/kathbath/kb_data_clean_wav"
# bucket_csv_base = "dataset/kathbath/kb_data_clean_wav"
save_path_base = "Results/Medium_125/kathbath_beam"
log_file = "Results/Medium_125/kathbath_beam/kathbath_all_languages_time_dummy.log"
wer_save_path = "Results/Medium_125/kathbath_beam/kathbath.txt"

prompt = False
# apply_lora = False

# Initialize log file
with open(log_file,'w') as log:
    log.write(f"Script started at {os.popen('date').read()}")

# Iterate over each language and run the Python script
for lang in languages:
    data_dir = f"{data_dir_base}/{lang}/test/audio"
    bucket_csv = f"{data_dir_base}/{lang}/test/bucket.csv"
    save_path = f"{save_path_base}/{lang}.csv"

    print(f"Running for language: {lang}")

    start_time = time.time()
    
    try:
        subprocess.run([
            "python", "whisper_inference.py",
            "--model_path", model_path,
            "--batch_size", str(batch_size),
            # "--do_normalize", do_normalize,
            "--language", lang,
            "--data_dir", data_dir,
            "--bucket_csv", bucket_csv,
            # "--chunk_size", str(chunk_size),
            "--save_path", save_path,
            "--wer_save_path", wer_save_path,
            # "--prompt",str(prompt),
            # "--apply_lora",str(apply_lora),
            # "--beam_search",str(beam_search)
        ], check=True)
        
        end_time = time.time()
        elapsed_time = (end_time - start_time)/60

        with open(log_file, 'a') as log:
            log.write(f"Completed run at {os.popen('date').read()}")
            log.write(f"Successfully completed for language: {lang}\n")
            log.write(f"Time taken for {lang}: {elapsed_time:.2f} minutes\n")

    except subprocess.CalledProcessError as e:
        # end_time = time.time()
        # elapsed_time = end_time - start_time

        with open(log_file, 'a') as log:
            log.write(f"Error running for language: {lang}\n")
            # log.write(f"Time taken before error for {lang}: {elapsed_time:.2f} seconds\n")


with open(log_file, 'a') as log:
    log.write(f"Completed run at {os.popen('date').read()}")
