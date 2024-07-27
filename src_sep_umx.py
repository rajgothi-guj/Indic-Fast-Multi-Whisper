import re
import os
import subprocess
import shutil

def split_audio_files(input_folder, chunk_length=600):
    """
    Splits all audio files in the specified folder into chunks of the specified length.

    :param input_folder: Path to the folder containing the audio files
    :param chunk_length: Length of each chunk in seconds (default is 600 seconds or 10 minutes)
    """
    if not os.path.isdir(input_folder):
        print(f"The specified path {input_folder} is not a directory.")
        return

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff')):
            input_filepath = os.path.join(input_folder, filename)
            file_base, file_ext = os.path.splitext(filename)
            split_folder = "/hdd2/Kumud/SPNI_Eval_Data/Avrodh_S01_1/Hindi/splited_wav"
            if not os.path.exists(split_folder):
                    os.makedirs(split_folder)
                
            output_pattern = os.path.join(split_folder, f"{file_base}_chunk%1n{file_ext}")

            # Construct the sox command
            command = [
                'sox', input_filepath, output_pattern,
                'trim', '0', str(chunk_length), ':', 'newfile', ':', 'restart'
            ]

            print(f"Splitting {filename} into chunks...")
            try:
                subprocess.run(command, check=True)
                print(f"Successfully split {filename} into 10-minute chunks.")
            except subprocess.CalledProcessError as e:
                print(f"An error occurred while splitting {filename}: {e}")

            # Process each chunk file for source separation
            output_folder = "/hdd2/Kumud/SPNI_Eval_Data/Avrodh_S01_1/Hindi/separated_audio"
            chunk_files = [f for f in os.listdir(split_folder) if f.startswith(file_base) and f.endswith(file_ext) and '_chunk' in f]
            chunk_files.sort(key=lambda f: int(re.search(r'_chunk(\d+)', f).group(1)))  # Sort files based on chunk number

            vocals_files = []
            for chunk_file in chunk_files:
                chunk_filepath = os.path.join(split_folder, chunk_file)
                chunk_output_dir = output_folder #os.path.join(output_folder, f"{chunk_file}_separated")
                
                if not os.path.exists(chunk_output_dir):
                    os.makedirs(chunk_output_dir)
                
                # Construct the umx command for source separation
                umx_command = [
                    'umx', chunk_filepath, '--outdir', chunk_output_dir
                ]
                
                print(f"Applying source separation on {chunk_file}...")
                try:
                    subprocess.run(umx_command, check=True)
                    print(f"Successfully separated sources for {chunk_file}.")
                    file_base1, file_ext = os.path.splitext(chunk_file)
            
                    vocals_filepath = os.path.join(chunk_output_dir,file_base1, 'vocals.wav')
                    print("vocals_filepath: ", vocals_filepath)
                    if os.path.exists(vocals_filepath):
                        vocals_files.append(vocals_filepath)
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred while separating sources for {chunk_file}: {e}")

            # Merge all vocals.wav files
            if vocals_files:
                final_merged_vocals_filepath = os.path.join('/hdd2/Kumud/SPNI_Eval_Data/Avrodh_S01_1/Hindi/seperate', f"{file_base}.wav")
                temp_merged_vocals_filepath = os.path.join(output_folder, f"{file_base}_temp_merged_vocals.wav")
                #final_merged_vocals_filepath = os.path.join(output_folder, f"{file_base}_merged_vocals.wav")
                merge_command = ['sox'] + vocals_files + [temp_merged_vocals_filepath]
                print(f"Merging vocals files into {temp_merged_vocals_filepath}...")
                try:
                    subprocess.run(merge_command, check=True)
                    print(f"Successfully merged vocals into {temp_merged_vocals_filepath}.")

                    # Convert the merged file to 16000 Hz, mono, 16-bit
                    convert_command = [
                        'ffmpeg', '-i', temp_merged_vocals_filepath,
                        '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
                        final_merged_vocals_filepath
                    ]
                    print(f"Converting {final_merged_vocals_filepath} to 16000 Hz, mono, 16-bit...")
                    try:
                        subprocess.run(convert_command, check=True)
                        # Remove output folder if it already exists
                        if os.path.exists(output_folder):
                            shutil.rmtree(output_folder)
                            shutil.rmtree(split_folder)
                        print(f"Successfully converted {final_merged_vocals_filepath} to 16000 Hz, mono, 16-bit.")
                    except subprocess.CalledProcessError as e:
                        print(f"An error occurred while converting {final_merged_vocals_filepath}: {e}")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred while merging vocals files: {e}")

# Example usage
input_folder = '/hdd2/Kumud/SPNI_Eval_Data/Avrodh_S01_1/Hindi/wav'
# split_folder = "/hdd5/kumud/spk_id/data/SPNI_Tarini_eval/Avrodh_S01/Hindi/splited_wav"
split_audio_files(input_folder, chunk_length=600)
