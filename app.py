import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import cv2
from skimage import io
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import layers, optimizers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from IPython.display import display
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, normalize
import os
import glob
import random
from google.colab import files #Libreria para cargar ficheros en Colab
%matplotlib inline

# Necesitamos montar su disco usando los siguientes comandos:
# Para obtener mas información sobre el montaje, puedes conusltar: https:
imagens = pd.read_jpg('../../js/images')

#Datos que contiene la ruta a Brian MRI y su mascara correspondiente
brain_df = pd.read_csv('../../js/csv/data_mask.csv')
brain_df.info()
brain_df.head(50)
brain_df.mask_path[1] #Ruta a la imagen de la MRI

brain_df.image_path[1] #Ruta a la mascara de segmentación

brain_df['mask'].value_counts()
brain_df

brain_df['mask'].value_counts().index
# Usaremos ploty para hacer un diagrama de barras interactivo
import plotly.graph_objects as go

fig = go.Figure([go.Bar(x=brain_df['mask'].value_counts().index, y = brain_df['mask'].value_counts())])
fig.update_traces(marker_color='rgb(0,200,0)', marker_line_color='rgb(0,255,0)',
                  marker_line_width=7, opacity=0.6)
fig.show()

brain_df.mask_path
brain_df.image_path

plt.imshow(cv2.imread(brain_df.mask_path[623]))
plt.imshow(cv2.imread(brain_df.image_path[623]))
cv2.imread(brain_df.mask_path[623]).max()
cv2.imread(brain_df.mask_path[623]).min()
# Visualización basica visualizaremos imágenes (MRI y mascaras) en el dataset de forma separarda
import random
fig, axs = plt.subplots(6,2, figsize=(16,32))
count = 0
for x in range(6):
  i = random.randint(0, len(brain_df)) # seleccionamos un indice aleatorio
  axs[count][0].title.set_text("MRI del cerebro") # Configuramos el título
  axs[count][0].imshow(cv2.imread(brain_df.image_path[i])) # Mostramos la MRI
  axs[count][1].title.set_text("Máscara -" + str(brain_df['mask'][i])) # Colocamos el titulo e
  axs[count][1].imshow(cv2.imread(brain_df.mask_path[i])) # Mostramos la máscara correspondiente
  count += 1

fig.tight_layout()
count = 0
fig,axs = plt.subplots(12,3, figsize=(20,50))
for x in range(len(brain_df)):
  if brain_df['mask'][x] == 1 and count <12:
    img = io.imread(brain_df.image_path[x]) # Load the original image, not the mask, for color overlay
    axs[count][0].title.set_text("MRI del cerebro")
    axs[count][0].imshow(img)

    mask = io.imread(brain_df.mask_path[x])
    axs[count][1].title.set_text('Mascara')
    axs[count][1].imshow(mask, cmap = 'gray')

    # Ensure 'img' has 3 channels (RGB) for color overlay:
    if img.ndim == 2:  # If 'img' is grayscale
        img = np.stack((img,)*3, axis=-1)  # Convert to RGB by stacking 3 copies

    img[mask == 255] = (255, 0, 0)  # Now you can overlay the color
    axs[count][2].title.set_text('MRI con Mascara')
    axs[count][2].imshow(img)
    count += 1

fig.tight_layout()
# Eliminamos la columnas de identificador del paciente
brain_df_train = brain_df.drop(columns = ['patient_id'])
brain_df_train.shape
# Convertir los datos en la columna de mascara a formato de string, para usar el modelo categorico en flow_from_dataframe
# Veras este mensaje de error si comentad la siguiente línea de código:
# TypeError: If class_mode="categorical", y_col="mask" column values must be type string, list or tuple.
brain_df_train['mask'] = brain_df_train['mask'].apply(lambda x: str(x))
brain_df_train.info()
# Dividir los datos en entrenamiento y testing
from sklearn.model_selection import train_test_split

train, test = train_test_split(brain_df_train, test_size = 0.15)
# Necesario en las nuevas versiones de Python en Colab
# Creamos el generador de imagenes
from keras_preprocessing.image import ImageDataGenerator

# Creamos un generador de datos que escale los datos de 0 a 1 y haga una división de validación de 0.15

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir ImageDataGenerator con validation_split
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)

# Generador de entrenamiento
train_generator = datagen.flow_from_dataframe(
    dataframe=train,
    directory='./',
    x_col="image_path",
    y_col="mask",
    subset='training',  # Solo funciona si datagen tiene validation_split
    batch_size=16,
    class_mode='categorical',
    target_size=(256,256)
)

# Generador de validación
valid_generator = datagen.flow_from_dataframe(
    dataframe=train,
    directory='./',
    x_col="image_path",
    y_col="mask",
    subset='validation',  # Solo funciona si datagen tiene validation_split
    batch_size=16,
    shuffle=True,
    class_mode='categorical',
    target_size=(256,256)
)

# Generador de prueba (sin validation_split)
test_datagen = ImageDataGenerator(rescale=1./255.)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test,
    directory='./',
    x_col="image_path",
    y_col="mask",
    batch_size=16,
    shuffle=True,
    class_mode='categorical',
    target_size=(256,256)
)
# Obtenemos el modelo base de ResNet50
basemodel = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(256,256,3)))
basemodel.summary()
# Congelamos los pesso del modelo
for layer in basemodel.layers:
  layer.trainable = False
  # Agregamos una cabecera de clasificacion al modelo base

headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size=(4,4))(headmodel)
headmodel = Flatten(name='flatten')(headmodel)
headmodel = Dense(256, activation='relu')(headmodel)
headmodel = Dropout(0.3)(headmodel)
#headmodel = Dense(256, activation = "relu")(headmodel)
#headmodel = Dropout(0.3)(headmodel)
headmodel = Dense(2, activation='softmax')(headmodel)

model = Model(inputs=basemodel.input, outputs=headmodel)
model.summary()
# compilamos el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Utilizamos la parada temprana para salir del entrenamiento si la perdida en la Validación no disminuye de ciertas epocas
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# Guardamos el mejor modelo con el menor perdida de validacion
checkpointer = ModelCheckpoint(filepath="classifier-resnet-weitghts.h5", verbose=1, save_best_only=True)
history = model.fit(train_generator,steps_per_epoch=train_generator.n // 16,
                    epochs = 1, validation_data = valid_generator,
                    validation_steps = valid_generator.n // 16,
                    callbacks=[earlystopping, checkpointer])