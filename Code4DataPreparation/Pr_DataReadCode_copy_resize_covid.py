# Import libraries
import pandas as pd
import numpy as np
import sys

# Read CVS file
data = pd.read_csv('./metadata.csv')

# Preprocess data (patients with no survival information are included)
# Only X-ray is selected
Data = data.copy()
Data.columns
Data = Data[Data['modality']=='X-ray']
Data = Data[Data['finding'].isin(['COVID-19','COVID-19, ARDS'])]
Data = Data[Data['view'].isin(['PA','AP', 'AP Supine'])]
Data = Data[['patientid','sex', 'age', 'survival','filename']]
# write data as csv
Data.to_csv('Data_cleaned.csv')

# Take the file name by indexing
DataFileName=np.array(Data.filename)
print (DataFileName[0])


# importing shutil module  
import shutil 
from skimage import io
from skimage import transform
import os
try:
    os.mkdir('resized_selected')
except:
    print('folder exists. removed.')        
    shutil.rmtree('resized_selected')
    os.mkdir('resized_selected')
    
for i in range(len(DataFileName)):
    # print(i)
    source = "./images/" + DataFileName[i]
    destination = "./resized_selected/" + DataFileName[i]
    IMG=io.imread(source)
    IMG=transform.resize(IMG, (512,512), anti_aliasing=True)
    io.imsave(destination,IMG)

    # dest = shutil.copyfile(source, destination) 



# Preprocess data (Only leave patients with survival information)
Data_survival = Data[Data['survival'].notna()]
# write data as csv
Data_survival.to_csv('Data_cleaned_survival.csv')

# Take the file name by indexing
Data_survivalFileName=np.array(Data_survival.filename)
print(Data_survivalFileName[0])

