# -*- coding: utf-8 -*-
"""
Created on Sat May 19 13:15:07 2018

@author: casa
"""
# =============================================================================
# import numpy
# from keras.preprocessing.image import ImageDataGenerator
# PRU_DIRECTORY = "./data/pru"
# TRAIN_DIRECTORY      = "./data/train"
# TEST_DIRECTORY       = "./data/test"
# VALIDATION_DIRECTORY = "./data/validation" 
# BATCH_SIZE = 32
# 
# 
# train_datagen = ImageDataGenerator(rescale=1./255)
# iterador = train_datagen.flow_from_directory (PRU_DIRECTORY, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=BATCH_SIZE, shuffle=True)
# iterador.next()
# =============================================================================

# =============================================================================
# (Test_x , Test_y) = ImageDataGenerator.flow_from_directory (
#         TEST_DIRECTORY, 
#         target_size=(256, 256), 
#         color_mode='rgb', 
#         classes=None, 
#         class_mode='categorical', 
#         batch_size=BATCH_SIZE, 
#         shuffle=True, seed=None, save_to_dir=None, save_prefix='', 
#         save_format='png', follow_links=False, subset=None, interpolation='nearest')
# =============================================================================

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
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

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')