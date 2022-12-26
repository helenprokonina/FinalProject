import os
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

from matplotlib import pyplot as plt

#import our custom classes
from utils import AudioUtil, SoundDS, ResBlock, ResNet18

parser = argparse.ArgumentParser()
parser.add_argument("--train", required=True)
parser.add_argument("--val", required=True)
parser.add_argument("--save_model", required=True)
args = parser.parse_args()




dataloader_train_path = args.train

dataloader_val_path = args.val

save_model_path = args.save_model


#dataloader_train_path = "dataloaders/dataloader_train.pth"
#dataloader_val_path = "dataloaders/dataloader_val.pth"
#save_model_path = "models/resnet18.pt"


# Create training and validation data loaders
train_dl = torch.load(dataloader_train_path)
val_dl = torch.load(dataloader_val_path)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# ----------------------------
# Training Loop
# ----------------------------
def train(model, train_dl, criterion, optimizer):
  # Loss Function, Optimizer and Scheduler

  print("Training")
  model.train()
  counter=0
  train_running_loss = 0.0

  correct_prediction = 0
  total_prediction = 0


    # Repeat for each batch in the training set
  for i, data in enumerate(train_dl):
      counter+=1
        # Get the input features and target labels, and put them on the GPU

      inputs, labels = data[0].to(device), data[1].to(device)

      labels.squeeze_()
        # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
      optimizer.zero_grad()


      outputs = model(inputs)

      # Get the predicted class probabilities
      outputs = torch.sigmoid(outputs)


      # Count of predictions that matched the target label
      correct_prediction += (outputs.argmax(-1) == labels.argmax(-1)).sum().item()
      total_prediction += labels.shape[0]

      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()


    # Keep stats for Loss and Accuracy
      train_running_loss += loss.item()



    # Print stats at the end of the epoch
  train_loss = train_running_loss / counter
  acc=correct_prediction/total_prediction
  print('Finished Training')
  return train_loss, acc




# ----------------------------
# Inference
# ----------------------------
def validate (model, test_dl, criterion):
  print("Validation")
  model.eval()

  counter = 0
  val_running_loss = 0.0

  correct_prediction = 0
  total_prediction = 0



  with torch.no_grad():
    for i, data in enumerate(test_dl):
      counter+=1
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      labels.squeeze_() #delete butch size

      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

# Get the predicted class probabilities
      outputs = torch.sigmoid(outputs)


      # Count of predictions that matched the target label
      correct_prediction += (outputs.argmax(-1) == labels.argmax(-1)).sum().item()
      total_prediction += labels.shape[0]

      loss = criterion(outputs, labels)
      val_running_loss += loss.item()




  val_loss = val_running_loss / counter
  acc=correct_prediction/total_prediction

  return val_loss, acc




num_epochs=20

model = ResNet18(2, ResBlock, outputs=11)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.BCELoss()


train_loss=[]
valid_loss=[]
acc_train=[]
acc_val=[]

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} of {num_epochs}")
    train_epoch_loss, train_epoch_acc = train(model, train_dl, criterion, optimizer)

    valid_epoch_loss, val_epoch_acc = validate(model, val_dl, criterion)

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    acc_train.append(train_epoch_acc)
    acc_val.append(val_epoch_acc)

    print(f"Train Loss: {train_epoch_loss:.4f}, Train acc: {train_epoch_acc: .4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}, Val acc: {val_epoch_acc: .4f}")



#save model
if not os.path.exists("models"):
    os.mkdir("models")
torch.save(model.state_dict(), save_model_path)

#----------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(range(num_epochs), train_loss, label='train_loss')
plt.plot(range(num_epochs), valid_loss, label='valid_loss')
plt.legend(labels=['train_loss','valid_loss'])
plt.title("Training/validation loss")

#save losses
if not os.path.exists("figures"):
    os.mkdir("figures")

plt.savefig('figures/train_val_losses.png')


#---------------------------------------------------------
plt.figure(figsize=(10,4))
plt.plot(range(num_epochs), acc_train, label='train_acc')
plt.plot(range(num_epochs), acc_val, label='valid_acc')
plt.legend(labels=['train_loss','valid_loss'])
plt.title("Training/validation accuracy")


#save accuracies
plt.savefig('figures/train_val_acc.png')

