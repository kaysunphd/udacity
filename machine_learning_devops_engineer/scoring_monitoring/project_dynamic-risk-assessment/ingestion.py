import pandas as pd
import numpy as np
import os
import pathlib
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

# create output folder
pathlib.Path(output_folder_path).mkdir(parents=True, exist_ok=True)

# today's date
today = datetime.today().strftime('%Y%m%d')

# save file with ingestion details
ingestion_filename = "ingestedfiles.txt"
ingested_files = []

# save merged data file
merged_filename = "finaldata.csv"
merged_filename = os.path.join(output_folder_path, merged_filename)

#############Function for data ingestion
def merge_multiple_dataframe():
    """
    check for datasets, compile them together, and write to an output file
    """
    if os.path.isfile(merged_filename):
        df_merged = pd.read_csv(merged_filename)
    else:    
        df_merged = pd.DataFrame()

    for root, subFolders, files in os.walk(input_folder_path):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.csv':
                filePath = os.path.join(root, file)
                df_file = pd.read_csv(filePath)
                df_merged = pd.concat([df_merged, df_file], axis=0)
                ingested_files.append([today, file])
    
    # de-dupe dataframe
    df_merged.drop_duplicates(inplace=True)

    # write to output csv
    df_merged.to_csv(merged_filename, index=False)

    # write to ingested data log
    with open(os.path.join(output_folder_path, ingestion_filename), 'a') as fp:
        for ingested in ingested_files:
            fp.write(", ".join(ingested))
            fp.write("\n")


if __name__ == '__main__':
    merge_multiple_dataframe()
