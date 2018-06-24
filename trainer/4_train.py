# -*- coding: utf-8 -*-
"""
Created on Sun Jun 3 2018

@author: casa
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import obtener_datos_entrenar
import numpy as np
import matplotlib as plt
import argparse


# parámetros para redimensionar imágenes
img_width, img_height = 150, 150 # ojo que está en otro componente
epochs = 50
batch_size = 64  
# con 16 - 176s/epoch (my_job18)
# con 128 - 200s/epoch  (job_3_1)
# con 64 - 176s/epoch (job_3_2)
# con 32 - 169s/epoch (job_3_3)
# con 96 - 162s/epoch (job_3_4)
# BASIC_GPU - 96 - 19s/epoch (job_3_5)
# BASIC_GPU - 128 - 20s/epoch (job_3_6)
# BASIC_GPU - 64 - ???s/epoch (job_3_7)

train_data_dir = ''
validation_data_dir = ''

classes_num = 128  
nb_train_samples = 12800
nb_validation_samples = 6105



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


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
    plt.Figure()
    
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
    
    (x_train, y_train, x_validation, y_validation) = obtener_datos_entrenar.obtener_datos(
            train_data_dir,validation_data_dir,classes_num)
    
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
###
###    PUNTO DE ENTRADA 
###
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
