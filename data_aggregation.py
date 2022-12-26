import os
from sys import argv
import argparse

import pandas as pd
import numpy as np
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import DataLoader, Dataset, random_split


import librosa
import librosa.display

#import our custom classes
from utils import AudioUtil, SoundDS, ResBlock, ResNet18



parser = argparse.ArgumentParser()
parser.add_argument("--download", required=True)
parser.add_argument("--train", required=True)
parser.add_argument("--save_train", required=True)
parser.add_argument("--save_val", required=True)
args = parser.parse_args()

#download_path = "IRMAS"
#train_path = "IRMAS-TrainingData"
#dataloader_train_path = "dataloaders/dataloader_train.pth"
#dataloader_val_path = "dataloaders/dataloader_val.pth"




download_path = args.download

train_path = args.train

dataloader_train_path = args.save_train

dataloader_val_path = args.save_val



folders = sorted(os.listdir(f"{download_path}/{train_path}/"))

classeID=['cello', 'clarinet', 'flute', 'acoustic guitar', 'electric guitar', 'organ',
          'piano', 'saxophone', 'trumpet', 'violin', 'voice']

labels_dict={'cel':0,
             'cla':1,
             'flu':2,
             'gac':3,
             'gel':4,
             'org':5,
             'pia':6,
             'sax':7,
             'tru':8,
             'vio':9,
             'voi':10}

inverse_dict={}
for key in labels_dict.keys():
    inverse_dict[labels_dict[key]]=key

num_classes=11

files_names=[]
labels_names=[]
num_channels=[]
sample_rates=[]
class_names=[]
#length of signals in seconds
sig_length=[]


#maximum value of amplitude
max_val=[]
#minimum value of amplitude
min_val=[]



folders = sorted(os.listdir(f"{download_path}/{train_path}/"))
for j, folder in enumerate(folders):
    #path=f"{download_path}/IRMAS-TrainingData/"+folder+"/"
    audio_files=os.listdir(f"{download_path}/{train_path}/"+folder+"/")
    for file_name in audio_files:
        files_names.append(f"/{train_path}/"+folder+"/"+file_name)
        labels=torch.zeros(1,num_classes)
        labels[0,j]=1
        labels_names.append(labels)

        class_names.append(inverse_dict[j])

        #open audio_file
        sig, sr = torchaudio.load(f"{download_path}/{train_path}/"+folder+"/"+file_name)
        #add number of channels in the list
        num_channels.append(sig.shape[0])


        sig_length.append(sig.shape[1]/sr)

        #signal in numpy
        sig_numpy=sig.detach().numpy()

        max_val.append(sig_numpy.max())
        min_val.append(sig_numpy.min())

        #add sample rate
        sample_rates.append(sr)



#create dataframe
df=pd.DataFrame({'relative_path': files_names, 'label': labels_names, 'num_channels': num_channels,
                 'sample_rates': sample_rates, 'time_duration': sig_length,
                 'min_val': min_val, 'max_val': max_val, 'class_name': class_names})



BATCH_SIZE=32


myds = SoundDS(df, download_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

#save dataloaders
if not os.path.exists("dataloaders"):
    os.mkdir("dataloaders")
torch.save(train_dl, dataloader_train_path)
torch.save(val_dl, dataloader_val_path)