# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 09:52:57 2022

@author: Alba
"""

"""
PROYECTO FINAL - VISIÓN POR COMPUTADOR

@author: Alba Casillas Rodríguez
@author: Jose Manuel Osuna Luque

*** INTRODUCCIÓN ***

El proyecto desarrollado se basa en el 2018 Data Science Bowl publicado por Kaggle:,
cuyo objetivo es detectar y segmentar núcleos celulares en un conjunto de imágenes ópticas
en distintas condiciones experimentales y con distintos protocolos de tinción sin la necesidad 
de ajustar manualmente los parámetros de la segmentación.

Se plantea abordar este problema mediante el uso de Mask R-CNN a partir de la implementación de código abierto
de matterport: https://github.com/matterport/Mask_RCNN/, basada en Python 3, Keras y Tensorflow. 
El modelo generará bounding boxes y máscaras de segmentación de las distintas instancias encontradas en una imagen.

*** INSTALACIÓN DEL ENTORNO ***

Para el correcto funcionamiento de la práctica se ha necesitado la instalación de un nuevo entorno:
    
        conda activate PROYECTOVC
        
Tras esto, se instala una versión más antigua de Python que será compatible con las versiones necesariaS
en el resto de herramientas.

    pip install python==3.7  --> version 3.7 (para poder instalar la version 1.13 de tensorflow la cual
                                ha sido droppeada a partir de python 3.8)
    
Las versiones de Tensorflow más recientes que son compatibles con los módulos utilizados por Mask-RCNN son:
    
    pip install tensorflow==1.13.1
    pip install tensorflow-gpu==1.15.0
    
Se instala CUDA para poder hacer uso de la GPU para disminuir el tiempo de ejecución del proyecto. 
Atendiendo a la documentacion: https://www.tensorflow.org/install/source#gpu ,
para que pueda ser compatible con la versión instalada de Tensorflow, será necesaria la versión 8, 
la cual se puede obtener a partir de: https://developer.nvidia.com/cuda-80-ga2-download-archive

NOTA: Si en el ordenador ya se encuentra una versión más reciente de CUDA, la más antigua no será reconocida en la ejecución. 
Por ello, una alternativa encontrada ha sido añadir a la carpeta bin de la versión más reciente de CUDA los archivos:
    
    
    - cudart64\_100.dll
    - nvcuda.dll
    - cublas64\_100.dll
    - cufft64\_100.dll
    - curand64\_100.dll
    - cusolver64\_100.dll
    - cuparse64\_100.dll
    - cudnn64\_7.dll

    
La siguiente instalación será la versión de Keras más nueva compatible con los módulos utilizados por el modelo.

    pip install keras==2.2.5


Para la lectura y procesamiento de los pesos, los cuales portan la extensión ".h5" será necesario:
    
    pip install h5py==2.10.0
    
Para evitar errrores de dependencias Spyder ha sido obligatorio realizar la instalación del módulo 
para control remoto paramiko.

    pip install paramiko==2.4.0

Por último, se han instalado librerías utilizadas durante el desarrollo de la práctica

    pip install mrcnn
    sudo install scikit-image
    pip install imgaug
    

NOTA: En Colab, tras la instalación es reiniciar el runtime (es decir, el entorno de ejecución).
"""


print("¡¡¡¡¡IMPORTANTE!!!!!")
print("ANTES DE CONTINUAR, LEA LAS INDICACIONES PROPORCIONADAS AL INICIO DEL PROGRAMA")
print("SE INDICARÁ PASO POR PASO EL ENTORNO VIRTUAL CREADO PARA QUE EL PROYECTO EJECUTE CORRECTAMENTE")
print("ADEMÁS, LAS CARPETAS QUE CONTIENEN LOS DATOS DEBEN ENCONTRARSE EN LA MISMA CARPETA DONDE ESTÁ EL EJECUTABLE")
print()

input("Pulse una tecla para continuar")

##############################################################################

# **LIBRERÍAS USADAS DURANTE LA PRÁCTICA**

##############################################################################

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Codiguillo para probar que uso las GPU y que no explota mi ordenador.

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))


# Libreria para operar con la entrada y salida de imagenes
# Es NECESARIA esta libreria ya que al utilizar mrcnn se usa internamente esto
# La falta de esta libreria causa el siguiente error: ModuleNotFoundError: No module named 'skimage'
import skimage.io
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

# Se importan las librerias necesarias para Mask RCNN
from mrcnn import utils, visualize
import mrcnn.model as modellib
from mrcnn.config import Config

# Importamos COCO
#import pycocotools.coco import COCO

# utils.download_trained_weights('/coco_weights.h5')

# Importamos otras librerias
import os
import cv2
import shutil
import keras
import random
import imgaug
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa
from mrcnn.model import log

import warnings
warnings.filterwarnings("ignore")

# Se carga la extensión de tensorboard (SOLO COLAB)
# %load_ext tensorboard

##############################################################################

# **FUNCIONES AUXILIARES**

##############################################################################

# Funcion que calcula STEPS_PER_EPOCH y VALIDATION_STEPS
# Doc: https://androidkt.com/how-to-set-steps-per-epoch-validation-steps-and-validation-split-in-kerass-fit-method/
# https://stackoverflow.com/questions/51748514/does-imagedatagenerator-add-more-images-to-my-dataset

def num_steps(data_size, batch_size):
    
    return data_size // batch_size


# De nuestro conjunto de datos train, realizamos una particion para obtener imagenes de validacion
# El porcentaje de validacion es establecido a un 10%

def particionSet(images_ids, split_validation=0.1):
    
    n_ids = len(images_ids)
    
    valid_size = int(n_ids * split_validation)
    
    valid_set = random.sample(images_ids, valid_size)
   
    train_set = list(set(images_ids) - set(valid_set))
    
    return train_set, valid_set


def particionSetSeleccion(images_ids, selection, split_validation=0.1):        
    valid_set = []
    valid_set.append(selection)
    
    train_set = list(set(images_ids) - set(valid_set))
    
    return train_set, valid_set


# Funcion que realiza la carga de pesos. Se diferencia entre los pesos obtenidos
# por los entrenamientos de los pesos de COCO ya que estos últimos necesitan que
# se exluden las ultimas capas porque requieren un numero de clases que coincida

def cargarWeights(modelo, name_w, WEIGHTS):
    
    if name_w.lower() == "coco":
        modelo.load_weights(WEIGHTS, by_name=True, exclude=["mrcnn_class_logits", 
                                            "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
    else:
        modelo.load_weights(WEIGHTS, by_name=True)
               
    return modelo

# Funcion que devuelve el generador para Data Augmentation

def getGenerador():
    
    # No se puede hacer uso de la clase ImageDataGenerator en este proyecto ya se obtiene el 
    # siguiente error: 
    # File "C:\Users\Alba\anaconda3\envs\PROYECTOVC\lib\site-packages\mrcnn\model.py", line 1257, in load_image_gt
    #    det = augmentation.to_deterministic()
    # AttributeError: 'ImageDataGenerator' object has no attribute 'to_deterministic'
    # donde, si investigamos la funcion load_image_gt, se necesita hacer uso de imagaug
    # generator = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, 
    #                               horizontal_flip=True, zoom_range = 0.2)
    
    # Con iaa.SomeOf se aplican algunas (entre 0 y 2) técnicas de Data Augmentation aleatoriamente
    # De entre las cuales encontraremos Volteos verticales y horizontales, transformaciones afines (rotaciones)
    # Zoom de entre [0.8 y 1.5] y desenfoque Gaussiano con un sigma de entre (0 - 5).
    # De todas las rotaciones, se utiliza iaa.OneOf para indicar que solo se puede realizar una de ellas a la vez.
      
    generador = iaa.SomeOf((0, 2), [ iaa.Fliplr(0.5), iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90), iaa.Affine(rotate=180),iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)), iaa.GaussianBlur(sigma=(0.0, 5.0)) ])
    
    return generador


# Funcion utilizada para visualizar distinttas imagenes en un mismo plot con un titulo asociado

def pintaConTitulo(vim, titles):
    num_im = len(vim)
    
    # Creamos un subplot por cada imagen
    # "_" sirve para ignorar el primer caracter (en este caso, la figura en sí)
    # Si no lo añadiesemos se obtendria el error: 'Figure' object has no attribute 'imshow'
    _, list_subplots = plt.subplots(1, num_im)  
    
    for i in range(num_im):
       list_subplots[i].imshow(vim[i], cmap="CMRmap") 
       list_subplots[i].set_title(titles[i])
    
       # list_subplots[i].xticks([]), list_subplots[i].yticks([])
       list_subplots[i].axis("off")
        
    plt.show()


"""

 ESTA FUNCIÓN SE ENCUENTRA COMENTADA PORQUE SOLO SE PUEDE USAR EN EL ENTORNO DE COLAB
 
 Es el código necesario para ejecutar la herramienta tensorboard, la cual se utiliza para
 poder mostrar la evolución de la función de pérdida para entrenamiento y validación de 
 forma gráfica.
 
 Esto no se puede realizar mediante código ya que, atendiendo a la implementación de Mask R-CNN
 no se guardan los valores de las funciones de pérdida en ninguna variable, solamente son guardados
 en los archivos que se generan en LOG_DIR
 
 !kill 729 -> A veces cuando se ejecuta %tensorboard no es posible mostrar los resultados porque el 
 proceso está ocupado (normalmente pasa cuando se hacen varias ejecuciones seguidas). El mensaje de error 
 de %tensorboard indicará que el proceso X está ocupado y es necesario terminarlo (kill),
 por tanto se escribe esta expresión y se añade el número del proceso (X, en este caso sería 729).
 Ambas sentencias deben estar en la misma celda de Colab
 
def ejecutarTensorboard():
    !kill 729
    %tensorboard --logdir="ruta/"
    
"""
    
    
# Funcion que calcula un todas las mascaras asociadas en una imagen
# en una unica mascara

def calcular_mascaras(mask, class_ids, limit=1):
          
    # Siempre será uno porque solo tenemos 1 clase
    unique_class_ids = np.unique(class_ids)    
           
    # El limite esta establecido a 1, por lo que v_masks tambien valdra 1
    # De esta forma solo se mostrara una única imagen que contiene todas las
    # mascaras solapadas
    for i in range(limit):
        
        if i < len(unique_class_ids):
            class_id = unique_class_ids[0]
        else:
            class_id = -1
        
        # Solo sera mascara aquella parte que tenga como class_id = 1. 
        # Donde sea -1 simplemente sera aquellas zonas donde no se encuentre mascara
        mascara = mask[:, :, np.where(class_ids == class_id)[0]]
        
        # Se realiza la suma de todas las mascaras de la imagen
        mascara = np.sum(mascara * np.arange(1, mascara.shape[-1] + 1), -1)
        

    return mascara
            
        
# Función que carga y visualiza un conjunto de imágenes aleatorias
    
def imagenes_aleatorias(dataset, num_img):
    
    samples_ids = np.random.choice(dataset.image_ids, num_img)

    for image_id in samples_ids:
        
        # Se carga la imagen
        image = dataset.load_image(image_id)
        
        # Se cargan las máscaras asociadas
        mask, class_ids = dataset.load_mask(image_id)
        
        # Se tiene en cuenta de que la variable "mask" contiene un vector de booleanos
        # y class_ids contiene un vector de unos con el numero de máscaras en la imagen
        # Por ello se debe calcular la localización (a modo numérico) de donde se 
        # encuentra cada una de las máscaras de la imagen
        mascara = calcular_mascaras(mask, class_ids)
        
        vim = [image, mascara]
        titles = ["imagen", "máscara"]
        
        pintaConTitulo(vim, titles)


# Funcion que calcula Average Precision (AP) dados los bounding boxes reales y predichos
     
def calculaAveragePrecision(gt_bbox, gt_class_id, gt_mask, result, porcentaje_iou):
    
    # Se calculan las correspondencias entre las predicciones y la verdadera localización de la máscara
    # El método devuelve:
    # gt_match = array 1D. Para cada Ground Truth Box, indica el indice de la predicción con la que ha hecho correspondencia.
    # pred_match = array 1D. Para cada predicción, indica el indice del ground truth box con el que ha hecho correspondencia.
    # overlaps = los solapamientos IoU (Intersection-over-Union) -> [pred_boxes,gt_boxes]
    # IoU calcula la proporción del área que forma la intersecciñon de las dos cajas frente al área de las dos cajas unidas.
    
    gt_match, pred_match, overlaps = utils.compute_matches(gt_bbox, gt_class_id, gt_mask, 
                                    result['rois'], result['class_ids'], result['scores'], result['masks'],
                                    iou_threshold=porcentaje_iou)
    
    # Para calcular el Average Precision (AP), se tiene que calcular la precision y el recall de cada caja de predicción
    # Precision = TP / TP + FP == TP / # predictions
    # Recall = TP / TP + FP = TP / # ground truths
    # Doc: https://hungsblog.de/en/technology/how-to-calculate-mean-average-precision-map/#confusion-matrix-tp-fp-fn
    # Doc: https://towardsdatascience.com/how-to-efficiently-implement-area-under-precision-recall-curve-pr-auc-a85872fd7f14
    
    precision = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    
    recall_suma = np.cumsum(pred_match > -1)
    # Se convierten los valores a float por si la division resulta en un valor decimal
    np.asarray(recall_suma, dtype=np.float)
    recall = recall_suma / len(gt_match)
    
    
    # Se añaden "valores centinelas" al principio y final para simplificar las matematicas
    precision_pad = np.concatenate(([0.], precision, [0.]))
    recall_pad = np.concatenate(([0.], recall, [1.]))
    
    # Se calcula Average Precision (AP)
    # El area bajo la curva de Precision Recall interpolada es el Average Precision
    # Doc: https://www.researchgate.net/publication/220659463_The_Pascal_Visual_Object_Classes_VOC_challenge
    # https://arxiv.org/pdf/1607.03476.pdf
    # https://github.com/rafaelpadilla/Object-Detection-Metrics  (buena documentacion)
    
    # Ahora se recorren los valores de precision en orden DECRECIENTE. De esta manera el valor de la precision
    # en comparacion con cada valor de recall sera siempre su valor maximo para todos los valores de recall siguientes
    for i in range((len(precision_pad) -1), 0, -1):
        # Se debe usar maximum en vez de max ya que con max se obtiene el siguiente error:
        # return umr_maximum(a, axis, None, out, keepdims, initial, where)
        # 'numpy.float64' object cannot be interpreted as an integer
        precision_pad[i-1] = np.maximum(precision_pad[i-1], precision_pad[i])
        
        # Para calcular el area bajo la curva Precision Recall, se deben de tener en cuenta
        # aquellos puntos donde el eje X (es decir, recall) cambia
        i = np.where(recall_pad[1:] != recall_pad[:-1])[0]
        average_precision = np.sum((recall_pad[i+1] - recall_pad[i]) * precision_pad[i+1])
    
    return average_precision, precision_pad, recall_pad


# Funcion que calcula el Average Precision para distintos umbrales de IoU y calcula el mean AP. 
def AveragePrecisionUmbrales(gt_bbox, gt_class_id, gt_mask, result):
    
    mAveragePrecision = []

    for umbral in UMBRALES_IOU:
        
        average_precision, precision_pad, recall_pad = calculaAveragePrecision(gt_bbox, gt_class_id, gt_mask, result, umbral)
        
        # print("AP con umbral: " + str(umbral) + "\t" + str(average_precision))
        print("AP con umbral: {:.2f}:\t {:.3f}".format(umbral, average_precision))
        
        mAveragePrecision.append(average_precision)

    mAP = np.array(mAveragePrecision).mean()

    #print("map[" + str(UMBRALES_IOU[0]) + "-" + (UMBRALES_IOU[-1]) + "]:" + str(mAP))
    print("mAP [{:.2f}-{:.2f}]: {:.3f}\t".format(UMBRALES_IOU[0], UMBRALES_IOU[-1], mAP))
    

            

##############################################################################

# **INICIALIZACIÓN DE VARIABLES GLOBALES**

##############################################################################

# Rutas que especifican las carpetas que almacenan las imagenes de train y test
RUTA_TRAIN = "stage1_train"
RUTA_PRUEBAS_LOCAL = "Pruebas"
RUTA_TEST = "stage1_test"


RUTA_ALBA_COLOR = "COLOR"

LOGS_LOCAL = "logs/"
LOGS_COLAB = '/content/drive/MyDrive/Images_ProyectoVC/logs/'

# PESOS DE COCO OBTENIDOS DE LA WEB: 
COCO_WEIGHTS_LOCAL = 'Pesos/mask_rcnn_coco.h5'

# Pesos obtenidos tras entrenar una única época partiendo de los pesos de COCO
# para adaptarlos a nuestro dataset.
# El entrenamiento estará realizado con imagenes de dimensión 128 y solamente
# se habrán entrenado las CABECERAS (HEADS)
NUCLEUS_WEIGHTS_1_EPOCH_HEADS = "Pesos/mask_rcnn_nucleus_0001.h5"

# Pesos obtenidos tras entrenar durante 20 épocas partiendo de los pesos de COCO
# El entrenamiento estará realizado con imágenes de dimensión 128 y solamente se habrán
# entrenado las cabeceras (HEADS)
NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_HEADS_NO_DAUG = "Pesos/mask_rcnn_nucleus_0020_128shape_heads_noaug.h5"

# Pesos obtenidos tras entrenar durante 20 épocas partiendo de los pesos de COCO
# El entrenamiento estará realizado con imágenes de dimensión 128 y solamente se habrán
# entrenado las cabeceras (HEADS). SE USA DATA AUGMENTATION.
NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_HEADS = "Pesos/mask_rcnn_nucleus_0020_128shape_heads.h5"

# Pesos obtenidos tras entrenar durante 20 épocas partiendo de los pesos de COCO
# El entrenamiento estará realizado con imágenes de dimensión 128 y solamente se habrá
# realizado un entrenamiento de TODA LA RED. SE USA DATA AUGMENTATION.
NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_ALL_LAYERS = "Pesos/mask_rcnn_nucleus_0020_128shape_all_layers.h5"

# Pesos obtenidos tras entrenar durante 20 épocas partiendo de los pesos de COCO
# El entrenamiento estará realizado con imágenes de dimensión 512 y solamente se habrán
# entrenado las cabeceras (HEADS). SE USA DATA AUGMENTATION.
NUCLEUS_WEIGHTS_20_EPOCHS_512SHAPE_HEADS = "Pesos/mask_rcnn_nucleus_0020_512shape_heads.h5"


# Corresponde al numero de imagenes usadas para el entrenamiento
SET_LEN = 670
# Porcentaje de validación
PORCENTAJE_VALIDACION = 0.1
# Estos dos parametros corresponden al número de épocas y tipo de entrenamiento
# que realizará el modelo
EPOCHS = 20
LAYERS = "heads"

TRAIN_SIZE = SET_LEN * (1 - PORCENTAJE_VALIDACION)
VALID_SIZE = SET_LEN * PORCENTAJE_VALIDACION

# Se crea un vector con distintos valorees para el umbral de IoU (para predecir si una caja es correcta)
UMBRALES_IOU = np.arange(0.5, 1.0, 0.05)



##############################################################################

# **CLASE NUCLEUSDATASET**

##############################################################################


# Clase necesaria para utilizar la clase 'Dataset' original (mrcnn/utils.py)   

"""
Según la documentación del código:
    Doc: https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/utils.py#L239
    
The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
"""

class NucleusDataset(utils.Dataset):

    """#########################################################################"""
    """             #FUNCIÓN PARA CARGAR EL CONJUNTO DE DATOS #                 """
    
    def cargar_set(self, dataset_dir, data_ids):
        
        # Se añade el número de clases que se encuentra en nuestro problema
        # En nuestro caso solamente será una, la clase nucleus
        self.add_class("nucleus", 1, "nucleus")

        for image_id in data_ids:

            ruta_imagen = dataset_dir + "/" + str(image_id) + "/images/{}.png".format(image_id)
            
            # Se guarda cada imagen en la clase Dataset. Para ello, se guarda
            # el id y la ruta (que nos permitira visualizar las imagenes).
            self.add_image("nucleus", image_id = image_id, path = ruta_imagen)

    """#########################################################################"""
    """            #FUNCIÓN PARA CARGAR LAS MASCARAS DE UNA IMAGEN #            """

    def load_mask(self, image_id):
        
        info = self.image_info[image_id]
        mask = []
        
        # info['path'] devuelve una estructura de dataset/id_image/images/id_image.png
        # Por lo que partimos la ruta delimitandolas por el caracter "/" para quedarnos
        # solamente con dataset/id_images y asi poder añadir que entre en la carpeta masks.
        ruta_sep = info['path'].split("/")
        mask_dir = ruta_sep[0] + "/" + ruta_sep[1] + "/masks"     
        
        list_mask = os.listdir(mask_dir)
        
        for m in list_mask:
            m_path = mask_dir + "/" + m
            m = skimage.io.imread(m_path).astype(np.bool)
            mask.append(m)
        mask = np.stack(mask, axis=-1)
        
        # Se devuelve el vector de máscaras y un vector de unos que indica el número
        # de máscaras que hay para una única imagen
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    """#########################################################################"""
    """           # FUNCIÓN QUE DEVUELVE LA REFERENCIA A LA IMAGEN #            """
    
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        
        if info["source"] == "nucleus":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
            

##############################################################################

# **CLASES DE CONFIGURACIÓN**

##############################################################################

# IMAGE SHAPE = 128
class NucleusMiniPyConfig(Config):

    NAME = "nucleus"

    # Batch size = GPU_COUNT * IMAGES_PER_GPU
    # A mayor batch_size, mas muestras seran propagadas
    # en la red neuronal y requiere calculos intermedios
    # mas grandes, que deben almacenarse en GPU. Por ello
    # mantendremos el valor a 1 y asi evitar problemas de memorias
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Se deben detectar los nucleus en las imagenes, por lo cual
    # nucleus es nuestra UNICA clase (+1 del background)
    NUM_CLASSES = 1 + 1  

    STEPS_PER_EPOCH = num_steps(TRAIN_SIZE,IMAGES_PER_GPU)
    
    VALIDATION_STEPS = num_steps(VALID_SIZE,IMAGES_PER_GPU)

    # Nivel de confianza. Los nucleos que se encuentren con
    # menos del 50% de confianza seran ignorados.
    DETECTION_MIN_CONFIDENCE = 0.5
    
    # En el paper original este valor es 100, pero lo aumentamos ya que
    # hay posibilidad de que para nuestro problema haya mas de 100 nucleos en una imagen
    DETECTION_MAX_INSTANCES = 500
    
    DETECTION_NMS_THRESHOLD = 0.7

    # Arquitectura de la red
    # Soporta resnet50 y resnet101 pero usaremos resnet50
    # ya que la hemos visto en clase y nos permitira entrenar
    # la red entera sin añadir una carga excesiva.
    BACKBONE = "resnet50"

    # 1040 x 1388 es la dimensión más grande que encontraremos
    # 256 x 256 es la dimensión más pequeña
    # Se ha decidido dejar todas las imagenes con un tamaño estandar
    # de 128 x 128. Para ello se usa el modo de resize "square", manteniendolo
    # de la configuración base, ya que redimensiona y añade padding de ceros
    # para conseguir el tamaño deseado.
    IMAGE_RESIZE_MODE = "square"
    
    # En el paper de Mask RCNN se menciona que el tamaño de la imagen
    # Paper: https://arxiv.org/pdf/1703.06870.pdf
    # se establece a 800 pero este valor genera el siguiente fallo:
    # Image size must be dividable by 2 at least 6 times to avoid fractions when downscaling and upscaling.
    # For example, use 256, 320, 384, 448, 512, ... etc.
    # Se elegiran inicialmente imágenes pequeñas para no añadir mucha carga en la ejecución.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Lista con las distintas dimensiones (en píxeles) del RPN
    # que localizará los distintos nucleus
    # Estos valores deben de ser divisible entre las dimensiones de la imagen
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)

    # Limita la cantidad de RoIS tras aplicar NMS. Se tienen menos para el
    # entrenamiento ya que de estos RoIs se elige un subconjunto para la siguiente
    # etapa del entrnamiento, y un valor muy alto podría afectar a la memoria.
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Cuantos anchors por imagen se van a utilizar para el entrenamiento
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Media RGB de todas las imagenes de train + validacion (strage_train)
    MEAN_PIXEL = np.array([43.516, 39.544, 48.208])

    # Numero de RoIs por imatgne que se utiliza para la clasificación
    TRAIN_ROIS_PER_IMAGE = 256

    RPN_NMS_THRESHOLD = 0.9

    # Maximo Numero de instancias de Ground Truth para usar en una imagen
    MAX_GT_INSTANCES = 500
    

# IMAGE SHAPE = 512
class NucleusPyConfig(Config):

    NAME = "nucleus"

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1  

    STEPS_PER_EPOCH = num_steps(TRAIN_SIZE,IMAGES_PER_GPU) 
    VALIDATION_STEPS = num_steps(TRAIN_SIZE,IMAGES_PER_GPU)


    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_MAX_INSTANCES = 500
    DETECTION_NMS_THRESHOLD = 0.7

    BACKBONE = "resnet50"

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    MEAN_PIXEL = np.array([43.516, 39.544, 48.208]) 

    TRAIN_ROIS_PER_IMAGE = 256
    
    RPN_NMS_THRESHOLD = 0.9

    MAX_GT_INSTANCES = 500

    
##############################################################################

# **CARGA DE DATOS. INICIALIZACION**

##############################################################################

def cargarDatos_train(RUTA):  
    # idImagenes_dir se recogen los id de todas las imagenes
    idImagenes_dir = os.listdir(RUTA)
    
    # Se realiza la partición en los conjuntos train y validaciñon
    train_ids, valid_ids = particionSet(idImagenes_dir, PORCENTAJE_VALIDACION)
    
    # Se carga el dataset para TRAIN
    dset_train = NucleusDataset()
    dset_train.cargar_set(RUTA, train_ids)
    dset_train.prepare()
    
    # Se carga el dataset para VALIDACIÓN
    dset_val = NucleusDataset()
    dset_val.cargar_set(RUTA, valid_ids)
    dset_val.prepare()
            
    return dset_train, dset_val


def cargarDatos_test(RUTA):
       
    # En este caso no hay división en la particion entre Train y Validation
    # porque toda la carpeta tiene solo imagenes de Test (que no se incluyen en Train ni Validation)
    idImagenes_dir = os.listdir(RUTA)
    
    dset_test = NucleusDataset()
    dset_test.cargar_set(RUTA, idImagenes_dir)
    dset_test.prepare()
               
    return dset_test


# Es un conjunto de datos con una única imagen

def cargar_seleccion(RUTA, id_seleccion):
           
    valid_set = []
    valid_set.append(id_seleccion)
        
    dset_seleccion = NucleusDataset()
    dset_seleccion.cargar_set(RUTA, valid_set)
    dset_seleccion.prepare()
    
    
    return dset_seleccion

##############################################################################

# **TRAIN MODEL**

##############################################################################


def trainModel(set_train, set_val, config_model, generator, name_w, WEIGHTS, LOGS):
  
    # Se crea el modelo
    # El argumento model_dir es OBLIGATORIO para que evitar errores para la creación del modelo
    # pero se borra al final. En esta carpeta se guardan los pesos del entrenamiento.
    modelo_rcnn = modellib.MaskRCNN(mode="training", config=config_model, model_dir=LOGS)
      
    modelo_rcnn = cargarWeights(modelo_rcnn, name_w, WEIGHTS)
    
    if generator == None:
        
        modelo_rcnn.train(set_train, set_val, learning_rate=config_model.LEARNING_RATE, epochs=EPOCHS, layers=LAYERS)
    else:
        
        modelo_rcnn.train(set_train, set_val, learning_rate=config_model.LEARNING_RATE, epochs=EPOCHS, 
                          augmentation=generator, layers=LAYERS)
 
    return modelo_rcnn

##############################################################################

# **PREDICTS. MODO INFERENCE**

##############################################################################

def detectionValidation(dset_val, config_model, name_w, WEIGHTS, LOGS):
    
    image_id = random.choice(dset_val.image_ids)
    
    modelo_detection = modellib.MaskRCNN(mode="inference", config=config_model, model_dir=LOGS)
    
    modelo_detection = cargarWeights(modelo_detection, name_w, WEIGHTS)
        
    # load_image_gt carga la información del "ground truth" (es decir, la localización de la máscara) 
    # de una imagen (imagen, mascara, bounding boxes)
    # Devuelve
    # image = la imagen cargada, dimensiones = metadatos de la imagen, es decir, las dimensiones de la imagen
    # gt_class_id = los class_id de la máscara (en nuestro caso, todos son 1), gt_bbox = devuelve bounding boxes
    # formados por [numero_instancia, (y1,x1,y2,x2)] y gt_mask = [height, width, numero_instancia] de la mascara
    image, dimensiones, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dset_val, config_model, image_id, use_mini_mask=False)

    print()
    print("ID: " + str(dset_val.image_reference(image_id)))
    print()
    
    results = modelo_detection.detect_molded(np.expand_dims(image, 0), np.expand_dims(dimensiones, 0), verbose=1)
    result = results[0] # results en un vector de 1 posicion
    
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    print()
    
    # Se muestra el resultado
    
    visualize.display_differences(image, gt_bbox, gt_class_id, gt_mask,
                                 result['rois'], result['class_ids'], result['scores'], result['masks'],
                                 dset_val.class_names, iou_threshold=0.95, score_threshold=0.95)
        
    return gt_bbox, gt_class_id, gt_mask, result



def detectionTest(dset_test, config_model, name_w, WEIGHTS, LOGS):

    model = modellib.MaskRCNN(mode="inference", config=config_model, model_dir=LOGS)
    
    model = cargarWeights(model, name_w, WEIGHTS)
        
    image_id = random.choice(dset_test.image_ids)
    
    img = dset_test.load_image(image_id)
    
    print()
    print("ID: " + str(dset_test.image_reference(image_id)))
    print()

     
    results = model.detect([img], verbose=1)

    result = results[0]    
        
    visualize.display_instances(img, result['rois'], result['masks'], result['class_ids'], dset_test.class_names, result['scores'],
                                show_bbox=False, show_mask=False)
 
    

##############################################################################

# **MAIN DEL PROGRAMA**

##############################################################################

print("----------- INICIO DEL PROGRAMA -----------")
print()
print("¡¡¡¡¡IMPORTANTE!!!!!")
print()
print("Cualquier entrenamiento ha sido comentado para evitar que tome demasiado tiempo la ejecución...")
print("¡Dentro del codigo se podrá encontrar todos los entrenamiento comentados!")
print("Todas las predicciones pueden hacerse calculando una imagen aleatoria" +
      " PERO se ha decidido usar un pequeño conjunto de imagenes para que se pueda ver la mejoria" +
      " de los entrenamientos tanto en la ejecución del código como en la memoria.")
print()

input("Pulse una tecla para continuar")
print()

print("----------- CARGANDO DATOS... -----------")
print()

dataset_train, dataset_val = cargarDatos_train(RUTA_TRAIN)

dataset_test = cargarDatos_test(RUTA_TEST)

# Se crea un conjunto de 4 imágenes de entrenamiento usadas para visualizar los resultados 
# de las distintas predicciones.
# Se han elegido a razón de su variabilidad entre ellas, para intentar reflejar en la mayor
# medida posible como el modelo detecta los núcleos en imágenes hechas en distintas condiciones.
conjunto_seleccionados = []

conjunto_seleccionados.append("5e263abff938acba1c0cff698261c7c00c23d7376e3ceacc3d5d4a655216b16d") 
conjunto_seleccionados.append("4590d7d47f521df62f3bcb0bf74d1bca861d94ade614d8afc912d1009d607b94") 
conjunto_seleccionados.append("d8607b21411c9c8ab532faaeba15f8818a92025897950f94ee4da4f74f53660a") 
conjunto_seleccionados.append("f29fd9c52e04403cd2c7d43b6fe2479292e53b2f61969d25256d2d2aca7c6a81") 

  
print("Se cargan algunas imagenes aleatorias del Train")

num_img = 4 # Numero de imagenes aleatorias a mostrar


imagenes_aleatorias(dataset_train, num_img)

input("Pulse una tecla para continuar")
print()

config_model = NucleusMiniPyConfig()

# El método display muestra un resumen de la configuración
# config_model.display()

print("----------- PRIMERA VERSIÓN -----------")
print()
print("Se entrena la CABECERA durante una época con los pesos de COCO -- [Llamada a la función comentada]")
print()

# EPOCHS = 1
# LAYERS = "heads"
# generador = None
# Resultado: NUCLEUS_WEIGHTS_1_EPOCH_HEADS

# trainModel(dataset_train, dataset_val, config_model, generador, "coco", COCO_WEIGHTS_LOCAL, LOGS_LOCAL)

print("Predicción y visualización de resultados")
print()

# Va a determinar si se elige una imagen en concreto para mostrar resultados o se muestra una aleatoria
# Para ello, si se elige que la imagen sea aleatoria, esta será obtenida del conjunto de validación.
# Por lo contrario, se utilizaran las imagenes de los id seleccionados.
# Esto se hace, sobre todo, de cara a la entrega de la práctica, de manera que se facilite poder visualizar
# las mejoras conseguidas con los distintos entrenamientos, tanto al ejecutar el código como en la memoria.

imagen_aleatoria = False

if imagen_aleatoria == True:
    
    gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_val, config_model, "other", NUCLEUS_WEIGHTS_1_EPOCH_HEADS, LOGS_LOCAL)  
else:
    
    for id_seleccion in conjunto_seleccionados:

        dataset_sel = cargar_seleccion(RUTA_TRAIN, id_seleccion)
        
        gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_sel, config_model, "other", NUCLEUS_WEIGHTS_1_EPOCH_HEADS, LOGS_LOCAL)
        
        print("Se calcula el Average Precision para distintos umbrales de IoU: ")
        print()
        
        AveragePrecisionUmbrales(gt_bbox, gt_class_id, gt_mask, result)
        
        input("Pulse una tecla para continuar")
        
        
               
dataset_sel = []

print()
print("----------- SEGUNDA VERSIÓN -----------")
print()
print("Se entrena la CABECERA durante 20 épocas con los pesos de COCO -- [Llamada a la función comentada]")
print()

# EPOCHS = 20
# LAYERS = "heads"
# generador = None
# Resultado: NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_HEADS_NO_DAUG

# trainModel(dataset_train, dataset_val, config_model, "coco", COCO_WEIGHTS_LOCAL, LOGS_LOCAL)

print("Predicción y visualización de resultados")
print()

if imagen_aleatoria == True:
    
    gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_val, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_HEADS_NO_DAUG, LOGS_LOCAL)  
else:
    
    for id_seleccion in conjunto_seleccionados:   
        
        dataset_sel = cargar_seleccion(RUTA_TRAIN, id_seleccion)
        
        gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_sel, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_HEADS_NO_DAUG, LOGS_LOCAL)
        
        print("Se calcula el Average Precision para distintos umbrales de IoU: ")
        print()
        
        AveragePrecisionUmbrales(gt_bbox, gt_class_id, gt_mask, result)
        
        input("Pulse una tecla para continuar")
        

        
dataset_sel = []       

print()
print("----------- TERCERA VERSIÓN -----------")
print()
print("Se entrena la CABECERA durante 20 épocas con los pesos de COCO y DATA AUGMENTATION -- [Llamada a la función comentada]")
print()


# EPOCHS = 20
# LAYERS = "heads"
# generador = getGenerador()
# Resultado: NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_HEADS

# trainModel(dataset_train, dataset_val, config_model, "coco", COCO_WEIGHTS_LOCAL, LOGS_LOCAL)

print("Predicción y visualización de resultados")
print()

if imagen_aleatoria == True:
    
    gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_val, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_HEADS, LOGS_LOCAL)  
else:
    
    for id_seleccion in conjunto_seleccionados:
        
        dataset_sel = cargar_seleccion(RUTA_TRAIN, id_seleccion)
        
        gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_sel, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_HEADS, LOGS_LOCAL)
        
        print("Se calcula el Average Precision para distintos umbrales de IoU: ")
        print()
        
        AveragePrecisionUmbrales(gt_bbox, gt_class_id, gt_mask, result)
        
        input("Pulse una tecla para continuar")
        
        
        
dataset_sel = []      

print()
print("----------- CUARTA VERSIÓN -----------")
print()
print("Se entrena TODA LA RED durante 20 épocas con los pesos de COCO y DATA AUGMENTATION -- [Llamada a la función comentada]")
print()


# EPOCHS = 20
# LAYERS = "all"
# generador = getGenerador()
# Resultado: NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_ALL_LAYERS

# trainModel(dataset_train, dataset_val, config_model, "coco", COCO_WEIGHTS_LOCAL, LOGS_LOCAL)

print("Predicción y visualización de resultados")
print()

if imagen_aleatoria == True:
    
    gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_val, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_ALL_LAYERS, LOGS_LOCAL)  
else:
    
    for id_seleccion in conjunto_seleccionados:
    
        dataset_sel = cargar_seleccion(RUTA_TRAIN, id_seleccion)
        
        gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_sel, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_128SHAPE_ALL_LAYERS, LOGS_LOCAL)
        
        print("Se calcula el Average Precision para distintos umbrales de IoU: ")
        print()
        
        AveragePrecisionUmbrales(gt_bbox, gt_class_id, gt_mask, result)
        
        input("Pulse una tecla para continuar")
        
        

dataset_sel = []      

print()
print("----------- QUINTA VERSIÓN -----------")
print()
print("Se entrena LA CABECERA durante 20 épocas con los pesos de COCO y DATA AUGMENTATION." +
       "Se aumenta las dimensiones de la imagen (512x12) -- [Llamada a la función comentada]")
print()

# se cambia la configuración del modelo a la que usa imagenes de 512x512
config_model = NucleusPyConfig()

# EPOCHS = 20
# LAYERS = "heads"
# generador = getGenerador()
# Resultado: NUCLEUS_WEIGHTS_20_EPOCHS_512SHAPE_HEADS

# trainModel(dataset_train, dataset_val, config_model, "coco", COCO_WEIGHTS_LOCAL, LOGS_LOCAL)

print("Predicción y visualización de resultados")
print()
imagen_aleatoria = True

if imagen_aleatoria == False:
      
    gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_val, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_512SHAPE_HEADS, LOGS_LOCAL)  

else:
       
    AP_imagenes = []
    prec_imagenes = []
    rec_imagenes = []
   
    for id_seleccion in conjunto_seleccionados:
    
        dataset_sel = cargar_seleccion(RUTA_TRAIN, id_seleccion)
        
        gt_bbox, gt_class_id, gt_mask, result = detectionValidation(dataset_sel, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_512SHAPE_HEADS, LOGS_LOCAL)
        
        print("Se calcula el Average Precision para distintos umbrales de IoU: ")
        print()
        
        AveragePrecisionUmbrales(gt_bbox, gt_class_id, gt_mask, result)
            
        input("Pulse una tecla para continuar")

print()       
print("Se realiza la detección de nucleos con imágenes aletorias del conjunto de test...")
print("Para ello se utilizan los pesos del ÚLTIMO entrenamiento realizado (mejores resultados)")
print()

for i in range(num_img):
    
    detectionTest(dataset_test, config_model, "other", NUCLEUS_WEIGHTS_20_EPOCHS_512SHAPE_HEADS, LOGS_LOCAL)
    
    input("Pulse una tecla para continuar")
    
    

# Borramos la carpeta que se ha tenido que crear obligatoriamente 
shutil.rmtree('logs/')