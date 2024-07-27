import os
import subprocess
import time

# List of language IDs
languages = ["hindi", "gujarati", "marathi", "bengali", "tamil", "telugu", "kannada", "malayalam"]
# languages = ["telugu"]

# Common parameters
model_path = "trained_model/Kathbath_fleurs_medium_FT"
batch_size = 32
beam_search = 5

save_path_base = "Results/Kathbath_fleurs_medium_FT/fleurs"
log_file = "Results/Kathbath_fleurs_medium_FT/fleurs/all_languages.log"
wer_save_path = "Results/Kathbath_fleurs_medium_FT/fleurs/wer.txt"

prompt = False

# Initialize log file
with open(log_file, 'w') as log:
    log.write(f"Script started at {os.popen('date').read()}")

# Iterate over each language and run the Python script
for lang in languages:
    # data_dir = f"{data_dir_base}/{lang}/test/audio"
    # bucket_csv = f"{bucket_csv_base}/{lang}/test/bucket.csv"
    save_path = f"{save_path_base}/{lang}.csv"

    print(f"Running for language: {lang}")

    start_time = time.time()
    try:
        subprocess.run([
            "python", "fleurs_whisper_inference.py",
            "--model_path", model_path,
            "--batch_size", str(batch_size),
            "--language", lang,
            "--save_path", save_path,
            "--wer_save_path", wer_save_path,
            "--split", 'test',
            # "--prompt",str(prompt),
            "--beam_search",str(beam_search)
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
