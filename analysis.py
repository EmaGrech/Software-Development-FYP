#Carrying out feature extraction

import cv2
import sys
import numpy as np
import pandas as pd
import torch 
import torchvision
import string, emoji, nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from collections import Counter
from keras.models import load_model
from torchvision.transforms import functional as F
from PIL import Image

##FETCHING NECESSARY DATA##
emojiScores = pd.read_csv('C:/Users/emagr/Documents/School/Y3S2/FYP/emoji/Emoji_Sentiment_Data_v1.0.csv') 
facialEmotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
objectList = {1:"person", 2:"bicycle", 3:"car", 4:"motorcycle", 5:"airplane", 6:"bus", 7:"train", 8:"truck", 9:"boat", 10:"traffic light", 11:"fire hydrant", 
12:"stop sign", 13:"parking meter", 14:"bench", 15:"bird", 16:"cat", 17:"dog", 18:"horse", 19:"sheep", 20:"cow", 21:"elephant", 22:"bear", 23:"zebra", 24:"giraffe", 
25:"backpack", 26:"umbrella", 27:"handbag", 28:"tie", 29:"suitcase", 30:"frisbee", 31:"skis", 32:"snowboard", 33:"sports ball", 34:"kite", 35:"baseball bat", 
36:"baseball glove", 37:"skateboard", 38:"surfboard", 39:"tennis racket", 40:"bottle", 41:"wine glass", 42:"cup", 43:"fork", 44:"knife", 45:"spoon", 46:"bowl", 
47:"banana", 48:"apple", 49:"sandwich", 50:"orange", 51:"broccoli", 52:"carrot", 53:"hot dog", 54:"pizza", 55:"donut", 56:"cake", 57:"chair", 58:"couch", 59:"potted plant", 
60: "bed", 61:"dining table", 62:"toilet", 63:"tv", 64:"laptop", 65:"mouse", 66:"remote", 67:"keyboard", 68:"cell phone", 69:"microwave", 70:"oven", 71:"toaster", 
72:"sink", 73:"refrigerator", 74:"book", 75:"clock", 76:"vase", 77:"scissors",78: "teddy bear", 79:"hair drier", 80:"toothbrush"}
facialDetecionModel = load_model('C:/Users/emagr/Documents/School/Y3S2/FYP/FER_model.h5')
objectDetecionModel = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

##WORDS#############################################
##TOKENISATION##
def tokenize (sentence):
    #Checking if the input is string (if it isn't, it is empty)
    if pd.isna(sentence) or not isinstance(sentence, str):
        sentence = ""
        return sentence
    else:
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        tokens = tokenizer.tokenize(sentence)
        return tokens

##CLEANING##
def clean(tokens):
    cleaned = []
    emojis = []
    stopWords = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    emojiChars = set(emoji.EMOJI_DATA)
    
    for token in tokens:
        newToken = ''
        i = 0
        
        while i < len(token):
            char = token[i]
            
            #Checking if emoji
            if char in emojiChars:
                emojis.append(char)
                i += 1
                
                #Isolates emoji if it is not seperate from another emoji/word
                if i < len(token) and token[i] != ' ':
                    break
            else:
                newToken += char
                i += 1
        
        #Processing tokens
        words = newToken.replace("’", "'").replace("'", "").split() #There was an issue with apostrophes
        posTags = nltk.pos_tag(words)
        for word, pos in posTags:
            #Remove non-alphabet, stop words, and empty strings
            word = word.translate(str.maketrans('', '', string.punctuation + '・')).strip()
            if word.isalpha() and word.lower() not in stopWords:                #Remove punctuation
                #Converting to lowercase
                word = word.lower()
                #Making POS tags readable by lemmatiser
                pos = pos[0].lower()  
                pos = pos if pos in ['a', 'r', 'n', 'v'] else 'n'  
                #Applying lemmatisation
                word = lemmatizer.lemmatize(word, pos)

                if word and word not in emojiChars: 
                    cleaned.append(word)

    return cleaned, emojis

##EMOJI SENTIMENT##
def emojiSentiment(emojis):
    total = 0

    if emojis:
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
    else:
        return 0

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

##TOKEN FREQUENCY##
def tokenFrequency(allTokens):
    newList = [token for sublist in allTokens for token in sublist]
    count = Counter(newList)
    
    #Calculating frequency
    freq = {token: count / len(allTokens) for token, count in count.items()}
    
    #Picking top 10
    top = dict(sorted(freq.items(), key=lambda item: item[1],reverse=True)[:10])

    return top

##N-GRAM FREQUENCY##
def nGramFrequency(allTokens):
    count = Counter()
    
    for tokens in allTokens:
        #Creating possible combinations
        nGrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens) - 2 + 1)]
        count.update(nGrams)

    #Calculating frequency
    total = sum(count.values())
    freq = {ngram: occur / total for ngram, occur in count.items()}
    
    #Picking top 10
    top = dict(sorted(freq.items(), key=lambda item: item[1],reverse=True)[:10])

    return top


##IMAGES#############################################
##COLOUR EXTRACTION##
def colourExtraction(image, bins=(8,8,8)):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #computing and normalising color histogram
    hist = cv2.calcHist([image_bgr], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist

##FACIAL EXTRACTION##
def facialExtraction(image):
    threshold = 0.5
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #Converting to greyscale
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Detecting faces
    faces = cascade.detectMultiScale(converted, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #Default if no face is found
    if len(faces) == 0:
        return 'No face detected / face obscured'

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
        
        if np.max(prob) > threshold:
            detected.append(predicted)

    return detected

def objectExtraction(image):
    threshold = 0.95
    objectDetecionModel.eval()

    #Converting image for model to be able to analyse
    if isinstance(image, np.ndarray):
        rgb = Image.fromarray(image.astype('uint8'), 'RGB')
    else:
        # If the image is a file path, open it
        rgb = Image.open(image).convert('RGB')
    tensorImage = F.to_tensor(rgb).unsqueeze(0)

    #Detecting objects
    with torch.no_grad():
        objs = objectDetecionModel(tensorImage)
        #Default if no object is found
        
    if len(objs) == 0:
        return 'No face detected / face obscured'
  
    labels = objs[0]['labels'].cpu().numpy()  
    scores = objs[0]['scores'].cpu().numpy()

    detected = [
        int(label) for label, score in zip(labels, scores)
        #Removing detected objects with low confidence
        if score > threshold
    ]
    
    detected = [objectList.get(label) for label in detected]

    #print("Labels:", labels)
    #print("Scores:", scores)

    return detected