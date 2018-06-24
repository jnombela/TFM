#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:02:32 2018

@author: justo
"""

import shutil, glob, os
import time
import argparse
import random


def crear_directorio(dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    
def copiar_ficheros(org,dest,clase,limite):
    files = glob.glob(org + '/*_' + clase + '.*')
    ##puede ser que no encuentre si ya hemos explorado todas las clases
    if len(files) == 0:
        return(0)
    ##comprobamos que existe
    crear_directorio(dest)
    ## mezclamos para obtener una muestra homogénea
    random.seed(42) # por reproducibilidad
    random.shuffle(files)
    ## aquí la copia
    copiados = 0
    for file in files:
        copiados +=1
        shutil.copy(file, dest)
        if (copiados >= limite):
            break
   
    print ("Copiados " + str(copiados) + " ficheros de la clase " + clase)
    return(copiados)
#################################################    
def repartir_carpetas_por_etiqueta(origen):   
    t_ini=time.time()
    total_ficheros = 0
    for i in range(250):
        part=str(i)
        total_ficheros += copiar_ficheros(origen, origen + '/' + part + '/', part,99999)
    
    print("--------------------------------------")
    print("---  TOTAL  " + str(total_ficheros) + "  ----")
    print("---  tiempo:" + str(time.time()- t_ini) + "----")
    print("--------------------------------------")

################################################
def copiar_hasta_limite(origen,destino,categorias,limite):   
    t_ini=time.time()
    total_ficheros = 0
    for i in categorias:
        part=str(i)
        total_ficheros += copiar_ficheros(origen, destino, part,limite)
    
    print("--------------------------------------")
    print("---  TOTAL  " + str(total_ficheros) + "  ----")
    print("---  tiempo:" + str(time.time()- t_ini) + "----")
    print("--------------------------------------")

##########################################
if __name__ == '__main__':
    # declaración de argumentos de entrada
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--origen", required=True, help="ruta de origen de los ficheros")
    ap.add_argument("-d", "--destino", required=False, help="ruta de destino")
    # parseo de argumentos
    args = vars(ap.parse_args())

    ##repartir_carpetas_por_etiqueta(out_dir)
    copiar_hasta_limite(args["origen"],args["destino"],[100],500)
    #copiar_hasta_limite(args["origen"],args["destino"],[42,43,88],100)
    