from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, LSTM

import process as ps

inputs, outputs = ps.main()

##SPLITTING##
xTrain, xTest, yTrain, yTest = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

##MODEL##
iLayer = Input(shape = (xTrain.shape[1], xTrain.shape[2]))
x = LSTM(128, return_sequences = True)(iLayer)
x = LSTM(64)(x)
oLayer = Dense(3, activation = 'softmax')(x)

model = Model(inputs = iLayer, outputs = oLayer)

##COMPILING##
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

##TRAINING##
model.fit(xTrain, yTrain, epochs = 10, batch_size = 32, validation_split = 0.2)

##SAVING MODEL##
model.save('Trained Model')

##EVALUATING##
loss, accuracy = model.evaluate(xTest, yTest)
print(f'Loss: {loss} \nAccuracy: {accuracy}')