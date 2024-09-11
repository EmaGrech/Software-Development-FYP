import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import numpy as np
import preparing as pr

filePath = 'C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx'
inputs, outputs = pr.main(filePath)

outputs = np.array(outputs)

if len(outputs.shape) == 2 and outputs.shape[1] > 1:
    y = np.argmax(outputs, axis=1)
else:
    y = outputs

##EVALUATING##
def eval(pred, yTest):
    acc = accuracy_score(yTest, pred)
    prec = precision_score(yTest, pred, average='weighted')
    recall = recall_score(yTest, pred, average='weighted')
    f1 = f1_score(yTest, pred, average='weighted')
    report = classification_report(yTest, pred)
    return acc, prec, recall, f1, report

##OUTPUTTING##
def outputResults(results, model_name):
    print(f"\n{model_name} Results:")
    for i, (acc, prec, recall, f1, training, pred, report) in enumerate(results):
        print(f"\nFold {i+1}:")
        print(f'Accuracy: {acc}')
        print(f'Precision: {prec}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')
        print(f'Training Time: {training} seconds')
        print(f'Prediction Time: {pred} seconds')
        print(f'Classification Report:\n{report}')

##LSTM##
def LSTMModel(xTrain, yTrain, xTest, yTest):
    #Building the model
    model = Sequential()
    model.add(LSTM(64, input_shape=(xTrain.shape[1], xTrain.shape[2]), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(len(np.unique(y)), activation='softmax')) 

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #Training the model
    start = time.time()
    model.fit(xTrain, yTrain, epochs=10, batch_size=32, validation_split=0.2)
    training = time.time() - start

    # Evaluating the model
    start = time.time()
    loss, acc = model.evaluate(xTest, yTest)
    pred = time.time() - start

    pred = np.argmax(model.predict(xTest), axis=-1)

    acc, prec, recall, f1, report = eval(pred, yTest)
    return acc, prec, recall, f1, training, pred, report

##Random Forest##
def RF(xTrain, yTrain, xTest, yTest):
    #Building the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    #Training the model
    start = time.time()
    model.fit(xTrain.reshape(xTrain.shape[0], -1), yTrain)
    training = time.time() - start

    #Evaluating the model
    start = time.time()
    pred = model.predict(xTest.reshape(xTest.shape[0], -1))
    pred = time.time() - start

    acc, prec, recall, f1, report = eval(pred, yTest)
    return acc, prec, recall, f1, training, pred, report

##Support Vector Machine ##
def SVM(xTrain, yTrain, xTest, yTest):
    #Building the model
    model = SVC()

    #Training the model
    start = time.time()
    model.fit(xTrain.reshape(xTrain.shape[0], -1), yTrain)
    training = time.time() - start

    #Evaluating the model
    start = time.time()
    pred = model.predict(xTest.reshape(xTest.shape[0], -1))
    pred = time.time() - start

    acc, prec, recall, f1, report = eval(pred, yTest)
    return acc, prec, recall, f1, training, pred, report

##Gradient Boosting Machine##
def GBM(xTrain, yTrain, xTest, yTest):
    #Building the model
    model = GradientBoostingClassifier()

    #Training the model
    start = time.time()
    model.fit(xTrain.reshape(xTrain.shape[0], -1), yTrain)
    training = time.time() - start

    #Evaluating the model
    start = time.time()
    pred = model.predict(xTest.reshape(xTest.shape[0], -1))
    pred = time.time() - start

    acc, prec, recall, f1, report = eval(pred, yTest)
    return acc, prec, recall, f1, training, pred, report

##RUNNING##
if __name__ == "__main__":
    #Assining number of folds
    splits = 5
    kf = KFold(n_splits=splits, shuffle=True, random_state=42)
    
    lstmResults = []
    rfResults = []
    svmResults = []
    gbmResults = []

    for train_index, test_index in kf.split(inputs):
        #Splitting the dataset
        xTrain, xTest = inputs[train_index], inputs[test_index]
        yTrain, yTest = y[train_index], y[test_index]
        yTrain = yTrain.flatten()
        yTest = yTest.flatten()

        print("Evaluating LSTM Model")
        lstmResults.append(LSTMModel(xTrain, yTrain, xTest, yTest))

        print("\nEvaluating Random Forest Model")
        rfResults.append(RF(xTrain, yTrain, xTest, yTest))

        print("\nEvaluating SVM Model")
        svmResults.append(SVM(xTrain, yTrain, xTest, yTest))

        print("\nEvaluating GBM Model")
        gbmResults.append(GBM(xTrain, yTrain, xTest, yTest))

    outputResults(lstmResults, "LSTM")
    outputResults(rfResults, "Random Forest")
    outputResults(svmResults, "SVM")
    outputResults(gbmResults, "GBM")