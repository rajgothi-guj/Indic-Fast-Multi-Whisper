def merge_files(file_names, output_path):
    merged_content = ""
    for file_name in file_names:
        with open(file_name, 'r') as file:
            merged_content += file.read()

    with open(output_path, 'w') as output_file:
        output_file.write(merged_content)

# Example usage:
language = ['hindi','gujarati','bengali','marathi','tamil','telugu','malayalam','kannada']

file_names = []
for i in range(8):
    file_names.append(f'preprocess_data/kathbath/{language[i]}/train.json')

# file_names = ['preprocess_data/hi/train.json', 'preprocess_data/gu/train.json']  # List of file names
output_path = 'preprocess_data/multilingual/merged_train.json'

merge_files(file_names, output_path)