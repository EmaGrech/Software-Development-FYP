from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np
import preparing as pr

filePath = 'C:/Users/emagr/Documents/School/Y3S2/FYP/FYP Statistics.xlsx'
inputs, outputs = pr.main(filePath)

outputs = np.array(outputs)

if len(outputs.shape) == 2 and outputs.shape[1] > 1:
    y = np.argmax(outputs, axis=1)
else:
    y = outputs

# SPLITTING DATASET##
xTrain, xTest, yTrain, yTest = train_test_split(inputs, y, test_size=0.2, random_state=42)

##LONG SHORT-TERM MEMORY##
def LSTMModel():
    #MODEL#
    model = Sequential()
    model.add(LSTM(64, input_shape=(xTrain.shape[1], xTrain.shape[2]), return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer size

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(xTrain, yTrain, epochs=10, batch_size=32, validation_split=0.2)

    #EVALUATING#
    loss, accuracy = model.evaluate(xTest, yTest)
    print(f'LSTM Accuracy: {accuracy}')

##RANDOM FOREST##
def RF():
    #MODEL#
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(xTrain.reshape(xTrain.shape[0], -1), yTrain)

    #EVALUATING#
    pred = model.predict(xTest.reshape(xTest.shape[0], -1))
    print(f'Random Forest Classification Report:\n{classification_report(yTest, pred)}')

##SUPPORT VECTOR MACHINE##
def SVM():
    #MODEL#
    model = SVC()
    model.fit(xTrain.reshape(xTrain.shape[0], -1), yTrain)

    #EVALUATING#
    pred = model.predict(xTest.reshape(xTest.shape[0], -1))
    print(f'Support Vector Machine Classification Report:\n{classification_report(yTest, pred)}')

##GRADIENT BOOSTING MACHINE##
def GBM():
    #MODEL#
    model = GradientBoostingClassifier()
    model.fit(xTrain.reshape(xTrain.shape[0], -1), yTrain)

    #EVALUATING#
    pred = model.predict(xTest.reshape(xTest.shape[0], -1))
    print(f'Gradient Boosting Machine Classification Report:\n{classification_report(yTest, pred)}')

##RUNNING##
if __name__ == "__main__":
    LSTMModel()
    RF()
    SVM()
    GBM()