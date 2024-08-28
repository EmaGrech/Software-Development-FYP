import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

    #FREQUENCY#
    for group, tokens in tFreqGrps.items():
        plotWords(tokens, group, "Token Frequency")
    
    for group, ngrams in nFreqGrps.items():
        plotWords(ngrams, group, "N-Gram Frequency")

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

##OTHER FUNCTIONS##
#FREQUENCY CONVERSION#
#Converting data to be usable for wordcloud
def convertFreq(toConvert):
    converted = []
    for item in toConvert.split(','):
        token, freq = item.split(':')
        token = token.strip()
        freq = float(freq.strip())
        converted.append((token, freq))
    
    return converted

##RUNNING##
if __name__ == "__main__":
    visualise()