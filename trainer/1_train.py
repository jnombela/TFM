# -*- coding: utf-8 -*-
"""
Created on Sat Jun 2 2018

@author: casa
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import glob
import random
import cv2

# parámetros para redimensionar imágenes
img_width, img_height = 150, 150

train_data_dir = 'short_data/train'
validation_data_dir = 'short_data/validation'
classes_num = 3  # numero de clases para clasificar
nb_train_samples = 4984
nb_validation_samples = 144

epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def leer_imagenes_etiquetas(ruta):
    ## variables
    x=[]
    y=[]

    ## obtenemos la lista de ficneros para la ruta pasada por parámentro
    filelist = sorted(glob.glob(ruta + "/*"))
    random.seed(42)
    random.shuffle(filelist)
    
    ## obtenemos las imágenes y las etiquetas de cada elemento de la lista
    for imagePath in filelist:
        	# lee, redimensiona y lo añade al array
        	imagen = cv2.imread(imagePath) # lee
        	imagen = cv2.resize(imagen, (img_width, img_height))  # redimentsiona
        	imagen = img_to_array(imagen) # pasa a array
        	x.append(imagen)
        
        	# extraemos la etiqueta del nombre de cada fichero
        	nombreimagen = imagePath.split("/")[2]   # le quitamos el resto de la ruta que no es el nombre
        	label = nombreimagen[nombreimagen.find("_")+1:nombreimagen.find(".")] # la etiqueta está entre _ y .
        	label = int(label) -1  # las etiquetas empiezan en 1, y les restamos 1 para que empiecen en 0
        	y.append(label)

#    x = np.array([np.array(Image.open(fname)) for fname in filelist])
#    ## obtenemos la lista de etiquetas y lo pasamos a One Hot Encoding con la ayuda de t_categorical
#    filenames = [fname.split("/")[2] for fname in filelist]
#    y = [(fname[fname.find("_")+1:fname.find(".")]) for fname in filenames]
#    y_cat=to_categorical(y)
    
    return (x,y)

def preproceso_imagenes_etiquetas(x,y):
    # pasar cada pixel a una escala de [0, 1] en un array numpy
    x = np.array(x, dtype="float") / 255.0
    # pasar las etiquetas a array y luego a categórico
    y = np.array(y)
    y_cat = to_categorical(y, num_classes=classes_num)
    
    return (x,y_cat)


def definir_red_neuronal():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    #model.add(Activation('sigmoid'))
    model.add(Dense(classes_num, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['categorical_accuracy'])

    return model

def definir_data_augmentation(x_train_p, y_train_p, x_validation_p, y_validation_p):
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    t_generator = train_datagen.flow(
        x_train_p,y_train_p,
        batch_size=batch_size)
    
    v_generator = test_datagen.flow(
        x_validation_p,y_validation_p,
        batch_size=batch_size)
    
    return (t_generator,v_generator)

###########################################################
###########################################################
###
###    PROGRAMA PRINCIPAL
###
###########################################################
###########################################################
    
## Leemos datos y etiquetas para train y validación, y los preprocesamos
(x_train, y_train) = leer_imagenes_etiquetas(train_data_dir)
(x_train, y_train) = preproceso_imagenes_etiquetas(x_train, y_train)
(x_validation, y_validation) = leer_imagenes_etiquetas(validation_data_dir)
(x_validation, y_validation) = preproceso_imagenes_etiquetas(x_validation, y_validation)

## definimos la estrategia de aumento de datos
(train_generator,validation_generator) = definir_data_augmentation(x_train, y_train,x_validation, y_validation)

## preparamos la arquitectura de la red neuronal
modelo = definir_red_neuronal()

## lanzamos el entrenamiento con los datos preparados y la red definida
modelo.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# guardamos el modelo entrenado
modelo.save_weights('first_try.h5')
