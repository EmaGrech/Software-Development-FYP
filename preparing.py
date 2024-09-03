#Preparing the Analysed data from excel to be readable as input for the neural network

import ast
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences

##INITIALISINIG##
filePath = "C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics 2.xlsx"

scaler = StandardScaler()
lEncoder = LabelEncoder()

##PARSING##
def parse(toParse, col):
    #Making the column contents parsable
    def convert(x):
        if isinstance(x, str):
            if x.startswith('[') and x.endswith(']'):
                try:
                    result = ast.literal_eval(x)
                    if not isinstance(result, list):
                        return [result]
                    return result
                except (SyntaxError, ValueError):
                    return [x]
            elif ':' in x:
                try:
                    pairs = x.split(', ')
                    result = {}
                    for pair in pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            result[key.strip()] = float(value.strip())
                    return result
                except (ValueError, TypeError):
                    return [x]
            else:
                return [x.strip()]
        else:
            return x if isinstance(x, list) else [x]

    #Parsing
    if col == 'Name':
        #Saving name as a string
        toParse[col] = toParse[col].apply(lambda x: str(x) if not isinstance(x, str) else x)
    else:
        toParse[col] = toParse[col].apply(lambda x: convert(x) if isinstance(x, str) else x)
    
    return toParse

##ENCODING##
def encode(toEncode, textCols):
    for col in textCols:
        if col in ['Emotions Detected in Image', 'Objects Detected in Image']:
            #Creating new individual columns for all emotions and objects
            expanded = toEncode[col].explode().reset_index(drop=True)
            #Encoding
            encoded = pd.get_dummies(expanded)
            toEncode = toEncode.drop(col, axis=1).join(encoded.groupby(level=0).sum(), how='left')
        elif col in ['Token Frequency', 'n-Gram Frequency']:
            #Creating new individual columns for all emotions and objects
            freqDataset = []
            for i, row in toEncode.iterrows():
                freqDict = row[col] if isinstance(row[col], dict) else {}
                freqData = pd.Series(freqDict, name=i).to_frame().T
                freqDataset.append(freqData)
            #Encoding
            combined = pd.concat(freqDataset).fillna(0)
            combined.columns = [f"{col}_{c}" for c in combined.columns]
            toEncode = toEncode.drop(col, axis=1).join(combined, how='left')
    
    return toEncode

##NORMALISING## 
def normalise(dataset, cols):
    #For handling low variance
    eps = 1e-8 

    for col in cols:
        #Replacing NaN and infinite values with 0
        dataset[col] = pd.to_numeric(dataset[col], errors='coerce')
        dataset[col] = dataset[col].replace([np.inf, -np.inf], np.nan)
        dataset[col] = dataset[col].fillna(0)
        
        #Prevent zero variance issues
        dataset[col] += eps
        
        #Scaling
        dataset[col] = scaler.fit_transform(dataset[[col]])
    
    return dataset

##SEQUENCING##
def sequence(dataset):
    inputs = []
    outputs = []
    
    cols = [col for col in dataset.columns if col not in ['Name', 'Label']]

    grouped = dataset.groupby('Name')

    for name, group in grouped:
        seq = group[cols].values.tolist()
        labels = group['Label'].values.tolist()

        inputs.append(np.array(seq, dtype='float32'))
        outputs.append(labels)
    
    #Converting into numpy arrays
    inputs = np.array(inputs, dtype=object)

    #Padding
    length = max(len(seq) for seq in inputs)
    padded = pad_sequences(inputs, maxlen = length, dtype = 'float32', padding = 'post')
    
    return padded, outputs

##RUNNING##
def main(filePath = 'C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx'):
    xls = pd.ExcelFile(filePath)
    sheets = xls.sheet_names

    toCombine = []
    textCol = ['Name', 'Emotions Detected in Image', 'Objects Detected in Image', 'Token Frequency', 'n-Gram Frequency']
    numCol = ['Token Sentiment Score', 'Emoji Sentiment Score', 'Image Colour Histogram']

    for sheet in sheets:
        toAdd = pd.read_excel(filePath, sheet_name = sheet, usecols = textCol + numCol)
        toAdd['Label'] = sheet

        for col in textCol:
            toAdd = parse(toAdd, col)

        toCombine.append(toAdd)
    
    dataset = pd.concat(toCombine, ignore_index = True)

    #Sending for encoding and normalisation
    dataset = encode(dataset, textCol)
    freqCols = [col for col in dataset.columns if col.startswith('Token Frequency_') or col.startswith('n-Gram Frequency_')]
    dataset = normalise(dataset, numCol + freqCols)
    
    #Assigning the inputs and outputs in sequences
    inputs, outputs = sequence(dataset)

    #Encoding outputs
    flat = [item for sublist in outputs for item in sublist]
    encoded = lEncoder.fit_transform(flat)
    encoded = encoded[:len(outputs)]
    encoded = np.array(encoded).reshape(len(outputs), -1) if len(outputs) > 0 else np.array([])