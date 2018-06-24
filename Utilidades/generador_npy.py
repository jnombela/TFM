#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:02:32 2018

@author: justo
"""

import argparse
import random
from tensorflow.python.lib.io import file_io 
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import io
import PIL
from google.cloud import storage
from tqdm import tqdm
import time

img_width, img_height = 150, 150


###################
## Leer todas las imágenes del disco
###################
def leer_imagenes_etiquetas(ruta):
    ## variables
    x=[]
    y=[]

    ## obtenemos la lista de ficneros para la ruta pasada por parámentro
    ruta=ruta + "/"
    print('LA RUTA ES...' + ruta)
    filelist = sorted(file_io.list_directory(ruta))
    random.seed(42)
    random.shuffle(filelist)

    ## obtenemos las imágenes y las etiquetas de cada elemento de la lista    
    print("EMPIEZA A LEER  " , time.time())
    
    with tqdm(total=len(filelist)) as t:
        for imgPath in filelist:
            imagen=leer_una_imagen(ruta + imgPath)
            imagen = img_to_array(imagen) # pasa a array
            x.append(imagen)
            
            label = imgPath[imgPath.find("_")+1:imgPath.find(".")] # la etiqueta está entre _ y .
            label = int(label) -1  # las etiquetas empiezan en 1, y les restamos 1 para que empiecen en 0
            y.append(label)
            t.update(1)
    print('REGISTROS LEIDOS X...' + str(len(x)))
    print('REGISTROS LEIDOS Y...' + str(len(y)))
    return (x,y)

###################
## Escalar las imágenes [0,1] y categorizar la salida
###################
def preproceso_imagenes_etiquetas(x,y,numero_clases):
    # pasar cada pixel a una escala de [0, 1] en un array numpy
    x = np.array(x, dtype="float") / 255.0
    # pasar las etiquetas a array y luego a categórico
    y = np.array(y)
    y_cat = to_categorical(y, num_classes=numero_clases)
    
    return (x,y_cat)

###################
## Leer una imagen del disco
###################
def leer_una_imagen(uri):
    image_bytes = file_io.FileIO(uri, mode='rb').read()
    image = PIL.Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((img_width, img_height), PIL.Image.BILINEAR)
    return image
#####################################
def obtener_rutas_locales(ruta):
        # obtenemos la ruta padre
    idx_ultima_barra=ruta.rfind('/')
    ruta_padre = ruta[0:idx_ultima_barra]
    # obtenemos el nombre del fichero a partir del nombre de la ultima carpeta
    troceado_ruta = ruta.split("/")
    nombre_fichero = troceado_ruta[len(troceado_ruta)-1]  
    # ruta completa con las dos partes
    rutaFichero_x=ruta_padre + '/' + nombre_fichero + 'x.npy'
    rutaFichero_y=ruta_padre + '/' + nombre_fichero + 'y.npy'
    
    return(rutaFichero_x,rutaFichero_y)
    
#################
def generar_fichero_npy(numero_clases):
    (x,y) = leer_imagenes_etiquetas(args["ruta_in"])
    (x,y) = preproceso_imagenes_etiquetas(x,y,numero_clases)
    (rutaLocal_x,rutaLocal_y) = obtener_rutas_locales(args["ruta_in"])
    print("EMPIEZA A GRABAR EN LOCAL", time.time())
    ##################
    # grabamos en local
    with file_io.FileIO(rutaLocal_x,'wb') as output:
        np.save(output,x)
        output.flush()
    with file_io.FileIO(rutaLocal_y,'wb') as output:
        np.save(output,y)
        output.flush()

    storage_client = storage.Client.from_service_account_json('/media/storage/proyectos/claves/Deep-1-cf9564fb42a8.json')
    bucket = storage_client.get_bucket('deep-1-203210-mlengine')        
    
    print("EMPIEZA A GRABAR EN BUCKET", time.time())
    ##################
    # y grabamos en cloud 
    blob=bucket.blob(args["ruta_out1"])
    with file_io.FileIO(rutaLocal_x,'rb') as leer:    
        blob.upload_from_file(leer)
        
    blob=bucket.blob(args["ruta_out2"])
    with file_io.FileIO(rutaLocal_y,'rb') as leer:    
        blob.upload_from_file(leer)



##########################################
if __name__ == '__main__':
    # declaración de argumentos de entrada
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ruta_in", required=True, help="ruta de origen de los ficheros")
    ap.add_argument("-1", "--ruta_out1", required=False, help="ruta de escritura 1")
    ap.add_argument("-2", "--ruta_out2", required=False, help="ruta de escritura 2")
    # parseo de argumentos
    args = vars(ap.parse_args())

    generar_fichero_npy(128)

    