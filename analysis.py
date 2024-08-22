import cv2
import sys
import numpy as np
import pandas as pd
import torch 
import string, emoji, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from collections import Counter
from keras.models import load_model
from PIL import Image

##FETCHING NECESSARY DATA##
emojiScores = pd.read_csv("C:/Users/emagr/Documents/School/Y3S2/FYP/emoji/Emoji_Sentiment_Data_v1.0.csv")

##TOKENISATION##
def tokenize (sentence):
    tokens = word_tokenize(sentence)
    return tokens

##CLEANING##
def clean(tokens):
    cleaned = []
    emojis = []
    stopWords = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    emoji_chars = set(emoji.emojiList.keys())
    
    for token in tokens:
        newToken = ""
        i = 0
        
        while i < len(token):
            char = token[i]
            
            #Checking if emoji
            if char in emoji_chars:
                emojis.append(char)
                i += 1
                
                #Isolates emoji if it is not seperate from another emoji/word
                if i < len(token) and token[i] != ' ':
                    break
            else:
                newToken += char
                i += 1
        
        #Processing tokens
        words = newToken.split()
        for word in words:
            #Remove numbers, stop words, and empty strings
            if word and not any(char.isdigit() for char in word) and word.lower() not in stopWords:
                #Remove punctuation
                word = word.translate(str.maketrans('', '', string.punctuation))
                #Converting to lowercase
                word = word.lower()
                #Applying stemming
                word = stemmer.stem(word)

                if word and word not in emoji_chars: 
                    cleaned.append(word)

    return cleaned, emojis

##EMOJI SENTIMENT##
def emojiSentiment(emojis):
    total = 0

    for emoji in emojis:
        #Converting emojis to HEX
        codepoint = hex(ord(emoji))
        emojiList = emojiScores[emojiScores['Unicode codepoint'] == codepoint]
        
        if not emojiList.empty:
            #Fetching data from csv
            positives = emojiList.iloc[0]['Positive']
            occurrences = emojiList.iloc[0]['Occurrences']
            
            if occurrences > 0:
                #Calculating how many of the occurrances were positive from the data  
                sentiment = positives / occurrences
            else:
                sentiment = 0  
            total += sentiment

    #Averaging     
    total = round(total/len(emojis), 4)
    return total

##TOKEN SENTIMENT##
def tokenSentiment(tokens):
    totalSentiment = 0
    totalNum = 0

    for token in tokens:
        #Fetching synonyms from WordNet
        synonyms = list(swn.senti_synsets(token))
        
        if not synonyms:
            continue

        #Calculating sentiment
        for synonym in synonyms:
            sentiment = synonym.pos_score() - synonym.neg_score()
            sentiment*= (1-synonym.obj_score())

            totalSentiment += sentiment
            totalNum += 1

    if totalNum > 0:
        #averaging
        average = round((totalSentiment / totalNum), 4)
        return average
    else:
        return 0
