#!/usr/bin/python3.5
# -*- coding:utf-8 -*-
# Created Time: Fri 02 Mar 2018 03:58:07 PM CST
# Purpose: download image
# Mail: tracyliang18@gmail.com
# Adapted to python 3 by Aloisio Dourado in Sun Mar 11 2018

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

import urllib3, os.path, time
from multiprocessing import Pool
from sys import argv,exit
from PIL import Image
from io import open as open_io
from io import BytesIO
from csv import writer as csvwriter
from tqdm import tqdm
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def definition(log_file):
    global logs 
    logs = open_io(log_file,"a")
    global logger 
    logger = csvwriter(logs,delimiter=',')
    # preparamos el logger para usar en todo el módulo   


def ParseData(data_file):

  ## para los que tienen etiquetas train y validation
  ann = {}
  if 'train' in data_file or 'validation' in data_file:
      _ann = json.load(open(data_file))['annotations']
      for a in _ann:
        ann[a['image_id']] = a['label_id']

  ## parseo de la información de las imágenes
  key_url_list = []
  j = json.load(open(data_file))
  images = j['images']
  for item in images:
    assert len(item['url']) == 1
    url = item['url'][0]
    id_ = item['image_id']
    if id_ in ann:
        id_ = "{}_{}".format(id_, ann[id_])
    key_url_list.append((id_, url))

  return key_url_list

################################
def DownloadImage(key_url):
  out_dir = argv[2]
  (key, url) = key_url
  filename = os.path.join(out_dir, '%s.jpg' % key)

  if os.path.exists(filename):
    ##print('Image %s already exists. Skipping download.' % filename)
    return

  try:
    #print('Trying to get %s.' % url)
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    image_data = response.data
  except:
    logger.writerow([1,'Warning: Could not download image %s from %s' % (key, url),key,url,filename])
    return

  try:
    pil_image = Image.open(BytesIO(image_data))
  except:
    logger.writerow([2,'Warning: Failed to parse image %s %s' % (key,url),key,url,filename])
    return

  try:
    pil_image_rgb = pil_image.convert('RGB')
  except:
    print()
    logger.writerow([3,'Warning: Failed to convert image %s to RGB' % key,key,url,filename])
    return

  try:
    pil_image_rgb.save(filename, format='JPEG', quality=90)
  except:
    logger.writerow([4,'Warning: Failed to save image %s' % filename,key,url,filename])
    return


def filter_key_url(list,class_filter):
    result=[]
    for item in list:
        (id_clase,URL) = item
        clase=id_clase.split("_")[1]
        for fil in class_filter:
            if (clase==fil):
                result.append(item)
    return(result)
        
     

def Run(data_file, out_dir,filter):
  key_url_list = ParseData(data_file)
  
  if (filter and 'test' not in data_file):
      lista = filter_key_url(key_url_list,filter)
  else:
      lista = key_url_list
  
  t1=time.time()
  p = Pool(processes=128)
    
  #p.imap_unordered(func=DownloadImage,iterable=key_url_list,chunksize=25)
  
  with tqdm(total=len(lista)) as t:
      for x in p.imap_unordered(func=DownloadImage,iterable=lista,chunksize=10):
        t.update(1)

  p.close()
  p.terminate()
  
  print("tiempo total:",time.time() - t1)
  
  
#  pool = multiprocessing.Pool(processes=12)
#  with tqdm(total=len(key_url_list)) as t:
#    for _ in pool.imap_unordered(DownloadImage, key_url_list):
#      t.update(1)

####################################
def comprobar_rutas(data_file, out_dir,log_file):
    
    if not os.path.exists(data_file):
        print('\n ¡¡¡no existe el fichero de datos!!! \n')
        exit(0)        
        
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)    
    
    # si no existe el fichero de logs, lo creamos
    if not os.path.exists(log_file):
        logs_aux = open_io(log_file,"x")
        logger_aux = csvwriter(logs_aux,delimiter=',')
        logger_aux.writerow(["Tipo","Texto","Clave_Imagen","URL","Nombre_Fichero"])
        logs_aux.close() 
 
##########################################
if __name__ == '__main__':
    if len(argv) != 3:
        print('Syntax: %s <train|validation|test.json> <output_dir/>' % argv[0])
        exit(0)     
    # preparamos las rutas y los logs
    (data_file, out_dir) = argv[1:]
    log_file = data_file+'.csv'
    comprobar_rutas(data_file, out_dir,log_file)
    definition(log_file)

    # lanzamos el proceso primcipal    
    #Run(data_file, out_dir,["42","43","88"])
    Run(data_file, out_dir,range(150))
    logs.close()
