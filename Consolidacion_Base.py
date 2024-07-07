#Importamos todas las librerias necesarias para el análisis
import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import missingno


#Montamos en Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Definimos la ruta a la carpeta que contiene los archivos CSV
folder_path = '/content/drive/My Drive/SAT Analisis de Datos/Ciudades/'

#Usamos la función glob para juntar todos los archivos CSV en la carpeta
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

#Armamos el Data Frame
dfs = []

for file in all_files:
    df = pd.read_csv(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)


#Valores faltantes, mostrar de todas las filas
pd.options.display.max_rows = None
pd.options.display.max_columns = None


#Vemos el tamaño del Dataset
combined_df.shape

#Vemos que columnas tienen, para ver que podríamos analizar
print(combined_df.columns)

combined_df.sample(5).transpose()

combined_df.info()

