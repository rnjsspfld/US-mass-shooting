### This script is authored by Hyerin Kwon
### When sharing, please ensure proper attribution to the original author.

# 1. speed-up concat processing
# import os
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
#
# # Path to the directory containing CSV files
# file_path = ''
#
# # Function to read CSV file and return DataFrame
# def read_csv(file_path):
#     try:
#         return pd.read_csv(file_path)
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return None
#
# # Initialize an empty list to store DataFrames
# dfs = []
#
# # Loop through each file in the directory
# with ThreadPoolExecutor() as executor:
#     # Collect futures
#     futures = []
#     for filename in os.listdir(file_path):
#         if filename.endswith('.csv'):
#             file_path2 = os.path.join(file_path, filename)
#             futures.append(executor.submit(read_csv, file_path2))
#
#     # Wait for all futures to complete
#     for future in futures:
#         df = future.result()
#         if df is not None:
#             dfs.append(df)
#
# # Concatenate all DataFrames in the list into a single DataFrame
# concatenated_df = pd.concat(dfs, ignore_index=True)
#
# # Display the concatenated DataFrame
# print(concatenated_df.shape)
#
# # save the file
# concatenated_df.to_csv("your_file_name.csv")

# 2. import csv file and save it as parquet file due to the memory issue
file_path = ''

chunk_size = 150000

chunks = []

for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
    print(f"Chunk {i + 1}: ", chunk)
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

print(df)

df.to_parquet('your_file_name.parquet')