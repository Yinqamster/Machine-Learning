from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import keras

model = Sequential()

model.add(Dense(units=512, input_dim=400))
model.add(Activation('relu'))
model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X = np.genfromtxt('train_data.csv',delimiter=',')
y = np.genfromtxt('train_targets.csv')
y = keras.utils.to_categorical(y,num_classes=10)

model.fit(X, y, epochs=10)

Xt = np.genfromtxt('test_data.csv',delimiter=',')
results = model.predict_classes(Xt)
np.savetxt('test_predictions_library.csv',results,newline='\n')
