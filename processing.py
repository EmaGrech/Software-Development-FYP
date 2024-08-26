import cv2
import sys
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import analysis as ay

##FETCHING##
dataset = pd.read_excel("C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx", sheet_name="Anxiety")

##PROCESSING##
def processing(row):
    tokenList = []
    tSentList = []
    emojiList = []
    eSentList = []
    histogramList = []
    faceList = []
    objList = []

    #Used for frequency calculations
    allTokens = []

    for name, group in dataset.groupby('Name'):
        allTokens.clear

        for index, row in dataset.iterrows():
            if index >= 5:  
                break
            
            print(f"\nProcessing row {index+1}/{len(dataset)}")
            caption = row['Caption']
            image = cv2.imread(row['Image Reference'])

            ##CAPTIONS##
            words = ay.tokenize(caption)
            cleaned, emojis = ay.clean(words)
            emojiSenti = ay.emojiSentiment(emojis)
            tokenSenti = ay.tokenSentiment(cleaned)

            allTokens.append(cleaned)

            ##IMAGES##
            colour = ay.colourExtraction(image)
            face = ay.facialExtraction(image)
            obj = ay.objectExtraction(image)

            ##APPENDING##
            tokenList.append(cleaned)
            tSentList.append(tokenSenti)
            emojiList.append(emojis)
            eSentList.append(emojiSenti)
            histogramList.append(colour)
            objList.append(obj)
            faceList.append(face)

            ##DEBUGGING##
            #print(f"Tokens: {words}")
            #print(f"Clean: {cleaned}")
            #print(f"Emojis: {emojis}")
            #print(f"Emoji Sentiment: {emojiSenti}")
            print(f"Caption Sentiment: {tokenSenti}")
            print(f"Colour: {colour}")
            print(f"Emotions: {face}")
            print(f"Objects: {obj}")
        
        ##FREQUENCIES##
        tFreq = ay.tokenFrequency(allTokens)
        nFreq = ay.nGramFrequency(allTokens)

        ##DEBUGGING##
        print(f"Token Freq {tFreq}")
        print(f"NGrams Freq: {nFreq}")
        sys.exit()

##SAVING##

##RUNNING##
if __name__ == "__main__":
    processing(dataset)