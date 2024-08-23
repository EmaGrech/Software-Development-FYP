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
facialEmotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
facialDetecionModel = load_model("C:/Users/emagr/Documents/School/Y3S2/FYP/FER_model.h5")

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
    
    emoji_chars = set(emoji.EMOJI_DATA)
    
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

##COLOUR EXTRACTION##
def colourExtraction(image, bins=(8,8,8)):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #computing and normalising color histogram
    hist = cv2.calcHist([image_bgr], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist

##FACIAL EXTRACTION##
def facialExtraction(image):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #Converting to greyscale
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Detecting faces
    faces = cascade.detectMultiScale(converted, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #Default if no face is found
    if len(faces) == 0:
        return "No face detected / face obscured"

    #Detecting emotions
    detected = []
    for (x, y, w, h) in faces:
        
        #Assigning Region Of Interest (ROI)
        roi = converted[y:y+h, x:x+w]
        roiResized = cv2.resize(roi, (48, 48))
        roiNormalized = roiResized / 255.0
        roiInput = np.reshape(roiNormalized, (1, 48, 48, 1))

        #Detecting emotions
        prob = facialDetecionModel.predict(roiInput)
        predicted = facialEmotions[np.argmax(prob)]
        
        detected.append(predicted)

    return detected