#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 20:33:42 2018

@author: justo
"""

from io import BytesIO
from tensorflow.python.lib.io import file_io
import numpy as np


###################
## Método principal
###################
def obtener_datos(train_data_dir,validation_data_dir,numero_clases):
    
    (x_train, y_train) = obtener_npy(train_data_dir)
    (x_validation, y_validation) = obtener_npy(validation_data_dir)

    return (x_train, y_train, x_validation, y_validation)

###################
## Rescatamos un fichero numpy 
###################
def obtener_npy(ruta):
    # obtenemos la ruta padre
    idx_ultima_barra=ruta.rfind('/')
    ruta_padre = ruta[0:idx_ultima_barra]
    # obtenemos el nombre del fichero a partir del nombre de la ultima carpeta
    troceado_ruta = ruta.split("/")
    nombre_fichero = troceado_ruta[len(troceado_ruta)-1]  
    # ruta completa con las dos partes
    rutaFichero_x=ruta_padre + '/' + nombre_fichero + 'x.npy'
    rutaFichero_y=ruta_padre + '/' + nombre_fichero + 'y.npy'
    
    print("RUTA X:   " + rutaFichero_x)
    print("RUTA Y:   " + rutaFichero_y)

    # parte principal. Si existe el pickle lo leemos, y si no, lo componemos
    if (file_io.file_exists(rutaFichero_x)):
        ## Rescatamos los datos que ya tenemos en un pickle
        print(" L E E M O S  NP ")
        with BytesIO(file_io.read_file_to_string(rutaFichero_x,binary_mode=True)) as input:
            x = np.load(input)
        with BytesIO(file_io.read_file_to_string(rutaFichero_y,binary_mode=True)) as input:
            y = np.load(input)
        return (x,y)
    else:
        exit(1)


  

