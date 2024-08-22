import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import analysis as ay

##FETCHING##
dataset = pd.read_excel("C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx", sheet_name="Anxiety")
row_index = 1
row = dataset.iloc[row_index]

##PROCESSING##
def processing(dataset):
    caption = row['Caption']

    words = ay.tokenize(caption)
    cleaned, emojis = ay.clean(words)
    emojiSenti = ay.emojiSentiment(emojis)
    tokenSenti = ay.tokenSentiment(cleaned)

    print(f"Tokens: {words}")
    print(f"Clean: {cleaned}")
    print(f"Emojis: {emojis}")
    print(f"Emoji Sentiment: {emojiSenti}")
    print(f"Caption Sentiment: {tokenSenti}")
    

##RUNNING##
if __name__ == "__main__":
    processing(dataset)