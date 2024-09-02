#Generator for visual representation of the data

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler

##FETCHING##
filePath = "C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx"
sheets = ['Anxiety', 'Depression', 'Both']

##VISUALISATION##
def visualise():
    
    #Initialising necessary variables
    colourGrps = {sheet: [] for sheet in sheets}
    tFreqGrps = {sheet: [] for sheet in sheets}
    nFreqGrps = {sheet: [] for sheet in sheets}
    tSentiGrps = {sheet: [] for sheet in sheets}
    eSentiGrps = {sheet: [] for sheet in sheets}

    #Populating variables
    for sheet in sheets:
        dataset = pd.read_excel(filePath, sheet)

        for _, row in dataset.iterrows():
            #FREQUENCY#
            tFreqData = row['Token Frequency']
            #Converting data to be usable for wordcloud
            if pd.notna(tFreqData):
                tConvert = convertFreq(tFreqData)
                tFreqGrps[sheet].extend(tConvert)

            nFreqData = row['n-Gram Frequency']
            #Converting data to be usable for wordcloud
            if pd.notna(nFreqData):
                nConvert = convertFreq(nFreqData)
                nFreqGrps[sheet].extend(nConvert)
            
            #SENTIMENT#
            tSentiData = row['Token Sentiment']
            tSentiGrps[sheet].append(tSentiData)

            eSentiData = row['Emoji Sentiment']
            eSentiGrps[sheet].append(eSentiData)

            #COLOUR#
            colourData = row['Colour']
            colourGrps[sheet].append(colourData)
    
    #FREQUENCY#
    for group, tokens in tFreqGrps.items():
        plotWords(tokens, group, "Token Frequency")
    
    for group, ngrams in nFreqGrps.items():
        plotWords(ngrams, group, "N-Gram Frequency")

    #SENTIMENT#
    plotSentiment(tSentiGrps, 'Token', True)
    plotSentiment(eSentiGrps, 'Emoji', False)

    #COLOUR#
    for group, hists in colourGrps.items():
        plotColour(hists, group) 

##INDIVIDUAL PLOTTING##
#FREQUENCY#
def plotWords(freq, group, colName):
    cloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
    ).generate_from_frequencies(dict(freq))

    plt.figure(figsize=(10, 8))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off') 
    plt.title(f'{colName}: {group} Group Dominant Tokens')
    plt.show()

#SENTIMENT#
def plotSentiment(sentiments, title, normal):
    
    sns.set_theme(style="whitegrid")
    df = convertSenti(sentiments, title, normal)
    yAxis = f'{title} Sentiment Scores'

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='group', y=yAxis, data = df)
    plt.title(f'Distribution of {title} Sentiment Scores Across Groups')
    plt.xlabel('Group')
    plt.ylabel(yAxis)

    plt.show()

#COLOUR#
def plotColour(colours, group):

    hists = convertColour(colours)
    colourAvg = np.mean(hists, axis=0)

    colourAvg *= 255
    colourAvg = colourAvg.astype('uint8')
    img = np.zeros((50, 300, 3), dtype=np.uint8)
    step = img.shape[1] // (len(colourAvg) // 3)
    
    for i in range(0, len(colourAvg), 3):
        colour = colourAvg[i:i+3]

        if len(colour) == 3:
            img[:, (i//3)*step:((i//3)+1)*step] = colour
    
    plt.figure(figsize=(5, 2))
    plt.axis('off')
    plt.title(f'{group} Group Dominant Colours')
    plt.imshow(img)
    plt.show()

##CONVERSION##
#FREQUENCY#
def convertFreq(toConvert):
    converted = []

    for item in toConvert.split(','):
        token, freq = item.split(':')
        token = token.strip()
        freq = float(freq.strip())
        converted.append((token, freq))
    
    return converted

#SENTIMENT#
def convertSenti(toConvert, sheet, normal):
    converted = []

    for group, scores in toConvert.items():
        for score in scores:
            converted.append({'group': group, f'{sheet} Sentiment Scores': score})

    df = pd.DataFrame(converted)
    
    if normal == True:
        scaler = MinMaxScaler()
        df[[f'{sheet} Sentiment Scores']] = scaler.fit_transform(df[[f'{sheet} Sentiment Scores']])
        
    return df

#COLOUR#
def convertColour(toConvert):
    converted = []

    for hist in toConvert:
        if isinstance(hist, str):
            cleaned = hist.replace('[', '').replace(']', '').strip()

            values = list(map(float, cleaned.split()))
            if values:
                    converted.append(values)
    
    return np.array(converted)

##RUNNING##
if __name__ == "__main__":
    visualise()