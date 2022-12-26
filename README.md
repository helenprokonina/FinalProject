# Classification of musical instruments in audio files

## Datasets

[IRMAS](https://www.upf.edu/web/mtg/irmas) dataset

### Training
IRMAS-TrainingData - dirrectory with folders, containing audiofiles for each instrument

### Testing
1. IRMAS-TestingData-Part1
2. IRMAS-TestingData-Part2
3. IRMAS-TestingData-Part3

each dirrectory contains .wav file and corresponding .txt file with the labels of instruments.

## Model
ResNet18 (custom)

## Jupyter notebook

#### instrument_classification_multi.ipynb

#### Description
This notebook contains the whole work from data aggregation from the Training folder till the classification of one-instrument records and inference on the data where more than one instrument can be found (Test folders).

## Python scripts

#### utils.py
contains all custom classes, necessary for correct work:
* AudioUtil for preprocessing audio files
* SoundDS - for preparing dataloaders
* ResNetBlock - block for ResNet18 model
* ResNet18 - model

#### data_aggregation.py
* collects the audio files from folders in the Test Dirrectory, 
* prepare dataframe from them, 
* makes a dataloaders for train and validation data
* saves them into dataloaders folder.

##### expected parameters (and examples):

```bash
data_aggregation.py --download "IRMAS" --train "IRMAS-TrainingData" --save_train "dataloaders/dataloader_train.pth" --save_val "dataloaders/dataloader_val.pth"
```

#### train.py
* imports dataloaders,
* trains the model ResNet18, 
* makes plots of training and validation losses and accuracies, 
* saves plots to the figures folder,
* saves the model into models folder.

##### expected parameters (and examples):

```bash
train.py --train "dataloaders/dataloader_train.pth" --val "dataloaders/dataloader_val.pth" --save_model  "models/resnet18.pt"
```

#### inference.py
* loads the model if it exists,
* prepare inference dataset from inference dirrectory, 
* choose 10 random records from the dataset, 
* predicts the instruments and shows the correct labels with the accuracy for a record.

##### expected parameters (and examples):

```bash
inference.py --download "IRMAS" --inference "/IRMAS-TestingData-Part2/IRTestingData-Part2/" --load_model  "models/resnet18.pt"
```

