# -*- coding: utf-8 -*-
"""
Created on Sun Jun 3 2018

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
from tensorflow.python.lib.io import file_io
import io
import PIL
import random
import matplotlib as plt
import argparse


# parámetros para redimensionar imágenes
img_width, img_height = 150, 150
epochs = 50
batch_size = 16

train_data_dir = ''
validation_data_dir = ''

classes_num = 3  # numero de clases para clasificar
nb_train_samples = 4984
nb_validation_samples = 144



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def leer_una_imagen(uri):
    image_bytes = file_io.FileIO(uri, mode='r').read()
    image = PIL.Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((img_width, img_height), PIL.Image.BILINEAR)
    return image


def leer_imagenes_etiquetas(ruta):
    ## variables
    x=[]
    y=[]

    ruta=ruta + "/"
    print('LA RUTA ES...' + ruta)
    ## obtenemos la lista de ficneros para la ruta pasada por parámentro
    filelist = sorted(file_io.list_directory(ruta))
    random.seed(42)
    random.shuffle(filelist)
    
    print('TAMAÑO DIRECTORIO...' + str(len(filelist)) + "  primero..." + filelist[1])
    ## obtenemos las imágenes y las etiquetas de cada elemento de la lista
    for imgPath in filelist:
        imagen=leer_una_imagen(ruta + imgPath)
        imagen = img_to_array(imagen) # pasa a array
        x.append(imagen)
        	# lee, redimensiona y lo añade al array
        
        #imagePath = ruta + imgPath
        #imagen = cv2.imread(imagePath) # lee
        #imagen = cv2.resize(imagen, (img_width, img_height))  # redimentsiona

        
        # extraemos la etiqueta del nombre de cada fichero
        #ruta_partes = imagePath.split("/")  # lo dividimos en las partes de la ruta   
        #nombreimagen = ruta_partes[len(ruta_partes)-1]   # nos quedamos con el nombre del fichero, que es el último
        
        label = imgPath[imgPath.find("_")+1:imgPath.find(".")] # la etiqueta está entre _ y .
        label = int(label) -1  # las etiquetas empiezan en 1, y les restamos 1 para que empiecen en 0
        y.append(label)
  
    print('REGISTROS LEIDOS X...' + str(len(x)))
    print('REGISTROS LEIDOS Y...' + str(len(y)))
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


def dibujar_grafico_accuracy_loss(H):
    plt.use("Agg")
    plt.use("ggplot")
    plt.figure()
    
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
    
    plt.title("Training Loss y Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["grafico"])


###########################################################
###########################################################
###
###    PROGRAMA PRINCIPAL
###
###########################################################
###########################################################

def Run():    
    ## Leemos datos y etiquetas para train y validación, y los preprocesamos
    (x_train, y_train) = leer_imagenes_etiquetas(train_data_dir)
    (x_train, y_train) = preproceso_imagenes_etiquetas(x_train, y_train)
    (x_validation, y_validation) = leer_imagenes_etiquetas(validation_data_dir)
    (x_validation, y_validation) = preproceso_imagenes_etiquetas(x_validation, y_validation)
    
    if (len(x_train)==0):
        print("ALGO HA IDO MAL!!!")
        return()
    ## definimos la estrategia de aumento de datos
    (train_generator,validation_generator) = definir_data_augmentation(x_train, y_train,x_validation, y_validation)
    
    ## preparamos la arquitectura de la red neuronal
    modelo = definir_red_neuronal()
    
    ## lanzamos el entrenamiento con los datos preparados y la red definida
    history = modelo.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
    
    # guardamos el modelo entrenado
    modelo.save_weights('first_try.h5')
    
    # guardamos la gráfica de accuracy y loss
    dibujar_grafico_accuracy_loss(history)

###########################################################
###########################################################
if __name__ == '__main__':
    # declaración de argumentos de entrada
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--rtrain", required=True, help="ruta de train")
    ap.add_argument("-v", "--rvalid", required=True, help="ruta de validation")
    ap.add_argument("-g", "--grafico", type=str, default="plot.png", help="ruta de grafico")
    # parseo de argumentos
    args = vars(ap.parse_args())

    # lanzamos el proceso primcipal   
    train_data_dir=args["rtrain"]
    validation_data_dir=args["rvalid"]
    Run()
