#Definicion de librerias con la funciones que seran utilizadas por Keras.
import keras
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

#Definicion de contenedor y primera capa de AlexNet.
print("Layer 1")
modelAlexNet = Sequential()
modelAlexNet.add(ZeroPadding2D((2,2), input_shape=(224, 224, 3)))
modelAlexNet.add(Convolution2D(96,11,11,subsample=(4,4),border_mode='valid'))
modelAlexNet.add(Activation(activation='relu'))
modelAlexNet.add(BatchNormalization())
print("Before Maxpooling-->", modelAlexNet.output_shape)
modelAlexNet.add(MaxPooling2D((3,3), strides=(2,2)))
print("After Maxpooling-->",modelAlexNet.output_shape)
print("#########")

#Definicion segunda capa
print("Layer 2")
modelAlexNet.add(ZeroPadding2D((2,2)))
modelAlexNet.add(Convolution2D(256,5,5,border_mode='valid'))
modelAlexNet.add(Activation(activation='relu'))
modelAlexNet.add(BatchNormalization())
print("Before Maxpooling-->", modelAlexNet.output_shape)
modelAlexNet.add(MaxPooling2D((3,3), strides=(2,2)))
print("After Maxpooling-->", modelAlexNet.output_shape)
print("#########")

#genera resumen de la red construida hasta el momento
print("Summary Layer 1 and 2")
modelAlexNet.summary()

#Definicion de tercera capa
print("Layer 3")
modelAlexNet.add(ZeroPadding2D((1,1)))
modelAlexNet.add(Convolution2D(384,3,3,border_mode='valid'))
modelAlexNet.add(Activation(activation='relu'))
print("Shape Layer 3-->",modelAlexNet.output_shape)

#genera resumen de la red construida hasta el momento
modelAlexNet.summary()

#Definicion de cuarta capa
modelAlexNet.add(ZeroPadding2D((1,1)))
modelAlexNet.add(Convolution2D(384,3,3,border_mode='valid'))
modelAlexNet.add(Activation(activation='relu'))
print(modelAlexNet.output_shape)

#Definicion de quinta capa
modelAlexNet.add(ZeroPadding2D((1,1)))
modelAlexNet.add(Convolution2D(256,3,3,border_mode='valid'))
modelAlexNet.add(Activation(activation='relu'))
modelAlexNet.add(MaxPooling2D((3,3), strides=(2,2)))
print(modelAlexNet.output_shape)

modelAlexNet.add(Flatten())
print(modelAlexNet.output_shape)

#Definicion de sexta capa
modelAlexNet.add(Dense(4096))
modelAlexNet.add(Activation(activation='relu'))
modelAlexNet.add(Dropout(0.5))
print(modelAlexNet.output_shape)

#Definicion de septima capa
modelAlexNet.add(Dense(4096))
modelAlexNet.add(Activation(activation='relu'))
modelAlexNet.add(Dropout(0.5))
print(modelAlexNet.output_shape)

#Definicion de octava capa
modelAlexNet.add(Dense(1000))
modelAlexNet.add(Activation('softmax'))
print(modelAlexNet.output_shape)

#genera resumen de la red construida hasta el momento
modelAlexNet.summary()