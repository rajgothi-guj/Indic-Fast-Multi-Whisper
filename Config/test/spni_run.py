import os
import subprocess
import time

# List of language IDs
# languages = [ "Hindi", "Marathi", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam"]
languages = [ "Hindi"]

# languages = [ "gujarati"]

# Common parameters
model_path = "trained_model/multi_large_baseline"
# model_path = "openai/whisper-large-v3"
batch_size = 32
do_normalize = False
# chunk_size = 

# Base directories and file names
bucket_csv_base = "/hdd2/Kumud/SPNI_Eval_Data/Avrodh_S01_1"
save_path_base = "Results/largev3_baseline/spni/avrodh_01/seperate"
log_file = "Results/largev3_baseline/spni/avrodh_01/seperate/spni_all_languages.log"
wer_save_path = "Results/largev3_baseline/spni/avrodh_01/seperate/spni_wer.txt"

# prompt = False

# Initialize log file
with open(log_file,'w') as log:
    log.write(f"Script started at {os.popen('date').read()}")

# Iterate over each language and run the Python script
for lang in languages:
    bucket_csv = f"{bucket_csv_base}/{lang}/bucket_sep.csv"
    save_path = f"{save_path_base}/{lang}.csv"

    print(f"Running for language: {lang}")

    start_time = time.time()
    
    try:
        subprocess.run([
            "python", "spni_inference.py",
            "--model_path", model_path,
            "--batch_size", str(batch_size),
            # "--do_normalize", do_normalize,
            "--language", lang.lower(),
            "--bucket_csv", bucket_csv,
            # "--chunk_size", str(chunk_size),
            "--save_path", save_path,
            "--wer_save_path", wer_save_path,
            # "--prompt",str(prompt),
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
