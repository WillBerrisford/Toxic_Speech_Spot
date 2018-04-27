import pandas as pd
import glob
import os
import dask.dataframe as dd

all_files_df = dd.read_csv('/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train_tagged/*.csv')
#file_locations = glob.glob(os.path.join(path, "*.csv"))

#print("Reading files")
#each_file_df = (pd.read_csv(f) for f in file_locations)

#print("Concatonating files")
#all_files_df = pd.concat(each_file_df, ignore_index=True)

#print("Writing data frames to csv")
ll_files_df.to_csv('/home/will/Computerscience/Machinelearning/Projects/Toxicspeechspot/Programdata/train_tagged.csv', index=False)
