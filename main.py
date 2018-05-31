import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.cluster import KMeans
import datetime
import collections

currentDT = datetime.datetime.now()
print (str(currentDT))

dataset = pd.read_csv('DatasetSGE_ValoresCorrectos.csv')

X = np.array(dataset[["25Fidelidad_TITULAR","16POBLACION_TITULAR","24Sexo_TITULAR", "15Motivos_Estancia_TITULAR", "23DISTANCIA_POBL_BALNE_TITULAR"]])
print(X.shape)

#####Seleccionar Columnas Dataset
#Poblacion y fidelidad
transactions = []

#####Seleccionar strings
for i in range(0, 11638):
    transactions.append([str(X[i,j]) for j in range (0, 5)])

#####Aplicando el algoritmo en el dataset
# Train Apriori Model
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2 , min_lift = 3)

# Visualising the results
results = list(rules)
myResults = [list(x) for x in results]
print("he acabado")


###Como funciona apriori
## https://www.youtube.com/watch?v=DtRIDCwXCt4
#support:
#
#confidence: = 0,2 (se repite un 20% )
#Aqui decimos el porcentaje que se va a repetir la compra ( si compra pan compra mantequilla se repite 20% en
#todas las transacciones)
#
#lift: lift of the relevance
#lift (confianza divido por el número de ejemplos cubiertos por la parte derecha de la regla),
# Nos sirve para detectar si existen reglas de asociacion o no