import cv2
import sys
import numpy as np
import pandas as pd
import torch 
import torchvision
import string, emoji, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
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
    #Checking if the input is string (if it isnt it is empty)
    if pd.isna(sentence) or not isinstance(sentence, str):
        sentence = ""
        return sentence
    else:
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
        newToken = ''
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
        return "No emojis used"

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
        
        detected.append(predicted)

    return detected

def objectExtraction(image):
    threshold = 0.5
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
        if score > 0.95
    ]
    
    detected = [objectList.get(label) for label in detected]

    print("Labels:", labels)
    print("Scores:", scores)

    return detected