import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import analysis as ay

##FETCHING##
dataset = pd.read_excel("C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx", sheet_name="Anxiety")

##PROCESSING##
def processing(row):
 
    caption = row['Caption']
    image = cv2.imread(row['Image Reference'])

    ##CAPTIONS##
    words = ay.tokenize(caption)
    cleaned, emojis = ay.clean(words)
    emojiSenti = ay.emojiSentiment(emojis)
    tokenSenti = ay.tokenSentiment(cleaned)

    ##IMAGES##
    colour = ay.colourExtraction(image)
    face = ay.facialExtraction(image)
    obj = ay.objectExtraction(image)

    ##DEBUGGING##
    print(f"Tokens: {words}")
    print(f"Clean: {cleaned}")
    print(f"Emojis: {emojis}")
    print(f"Emoji Sentiment: {emojiSenti}")
    print(f"Caption Sentiment: {tokenSenti}")
    print(f"Colour: {colour}")
    print(f"Emotions: {face}")
    print(f"Objects: {obj}")

##RUNNING##
if __name__ == "__main__":
    for index, row in dataset.iterrows():
        print(f"\nProcessing row {index+1}/{len(dataset)}")
        processing(row)