#Generator for visual representation of the data

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler

##FETCHING##
filePath = "C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx"
sheets = ['Anxiety', 'Depression', 'Both', 'Neither']

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
            tSentiData = row['Token Sentiment Score']
            tSentiGrps[sheet].append(tSentiData)

            eSentiData = row['Emoji Sentiment Score']
            eSentiGrps[sheet].append(eSentiData)

            #COLOUR#
            colourData = row['Image Colour Histogram']
            colourGrps[sheet].append(colourData)
    
    #FREQUENCY#
    for group, tokens in tFreqGrps.items():
        plotWords(tokens, group, "Token Frequency")
    
    for group, ngrams in nFreqGrps.items():
        plotWords(ngrams, group, "n-Gram Frequency")

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

    plt.figure(figsize=(10, 4))
    channels = ['Red', 'Green', 'Blue']
    colors = ['r', 'g', 'b']

    for i, channel in enumerate(channels):
        plt.plot(colourAvg[i::3], color=colors[i], label=f'{channel} Channel')

    plt.xlabel('Color Bin Index')
    plt.ylabel('Normalized Frequency')
    plt.title(f'{group} Group Dominant Colours')
    plt.legend()
    plt.grid(True)
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
            
            # Debug: Print the cleaned string
            
            values = list(map(float, cleaned.split()))
            if values:
                converted.append(values)
    
    # Convert to numpy array and print shape and sample values
    converted_array = np.array(converted)
    print(f"Converted Array Shape: {converted_array.shape}")
    print(f"Sample Converted Array: {converted_array[:1]}")
    
    return converted_array


##RUNNING##
if __name__ == "__main__":
    visualise()