import os
import sys
from sys import argv
import argparse

import pandas as pd
import numpy as np


import random

import torch
from torch import nn
import torchaudio
from torchaudio import transforms
from torch.utils.data import DataLoader, Dataset, random_split


import librosa
import librosa.display

#import our custom classes
from utils import AudioUtil, SoundDS, ResBlock, ResNet18

parser = argparse.ArgumentParser()
parser.add_argument("--download", required=True)
parser.add_argument("--inference", required=True)
parser.add_argument("--load_model", required=True)
args = parser.parse_args()


download_path = args.download

inference_path = args.inference

model_load_path = args.load_model



#inference_path = "/IRMAS-TestingData-Part2/IRTestingData-Part2/"
#download_path = "IRMAS"
#load_model_path = "models/resnet18.pt"


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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = ResNet18(2, ResBlock, outputs=11)

#load model weights
if os.path.exists(model_load_path):
    model.load_state_dict(torch.load(model_load_path),strict=False)
else:
    print("Error! There is no model available")
    sys.exit()

model.eval()


inf_audio=[]
inf_labels=[]
inf_files=os.listdir(f"{download_path}{inference_path}")
for file_name in inf_files:
        if "txt" in file_name:
            with open(f"{download_path}{inference_path}"+file_name, encoding = 'utf-8') as f:
                contents = f.read().splitlines()
                labels=[]
                for line in contents:
                    labels.append(labels_dict[line[:-1]])
            one_hot=torch.zeros(1,num_classes) #one-hot encoding for classes [1,1,0,0,....] = cel + cla
            one_hot[0,labels]=1
            inf_labels.append(one_hot)
        elif "wav" in file_name:
            inf_audio.append(inference_path+file_name)


inf_df=pd.DataFrame({'relative_path': inf_audio, 'label': inf_labels})


#choose 10 random records from TestData-Part3
num=10

indexes=np.random.choice(range(inf_df.shape[0]), size=num)
print(indexes)

indexed_df=inf_df.iloc[indexes]

indexed_df.reset_index(inplace=True, drop=True)





inf_ds = SoundDS(indexed_df, download_path)

print(f"size of validation data set {len(inf_ds)}")
print("-"*30)
print("\n")


# Create training and validation data loaders
inf_dl = torch.utils.data.DataLoader(inf_ds, batch_size=1, shuffle=False)





for j, data in enumerate(inf_dl):
    inputs, labels = data[0].to(device), data[1].to(device)

    labels.squeeze_() #delete butch size
      # Normalize the inputs
    inputs_m, inputs_s = inputs.mean(), inputs.std()
    inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
    outputs = model(inputs)

# Get the predicted class probabilities
    outputs = torch.sigmoid(outputs)

    outputs = outputs.detach().numpy()



    target_indices = [i for i in range(len(labels)) if labels[i] == 1]

    sorted_indices = np.argsort(outputs[0])
    best = sorted_indices[-len(target_indices):] #take the biggest probabilities the same number as target instruments
    predicted=np.zeros_like(labels)
    predicted[best]=1



    target_set=set(target_indices)
    pred_set=set(best)


    acc=len(target_set.intersection(pred_set))/len(target_set)


    print(f"accuracy for a record {indexes[j]}: {acc:.2f}")



    string_predicted = ''
    string_actual = ''
    for i in range(len(best)):
        string_predicted += f"{inverse_dict[best[i]]}   "
    for i in range(len(target_indices)):
        string_actual += f"{inverse_dict[target_indices[i]]}  "


    print(f"actual instruments: {string_actual}")
    print(f"predicted instruments: {string_predicted}")
    print("-"*30)
    print("\n")











