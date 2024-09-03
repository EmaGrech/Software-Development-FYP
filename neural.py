#The machine learning algorithm

#for Keras
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, LSTM

#for PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#for Hugging Face
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

import preparing as ps

filePath = 'C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx'
inputs, outputs = ps.main(filePath)

##SPLITTING##
xTrain, xTest, yTrain, yTest = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

##KERAS MODEL##
def kModel():
    #MODEL#
    iLayer = Input(shape = (xTrain.shape[1], xTrain.shape[2]))
    x = LSTM(128, return_sequences = True)(iLayer)
    x = LSTM(64)(x)
    oLayer = Dense(3, activation = 'softmax')(x)

    model = Model(inputs = iLayer, outputs = oLayer)

    #COMPILING#
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    #TRAINING#
    model.fit(xTrain, yTrain, epochs = 10, batch_size = 32, validation_split = 0.2)

    #EVALUATING#
    loss, accuracy = model.evaluate(xTest, yTest)
    print(f'Loss: {loss} \nAccuracy: {accuracy}')

##PYTORCH MODEL##
def pModel():



##HUGGING FACE TRANSFORMER##

##RUNNING ALL##
#if __name__ == "__main__":
