import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from mpl_toolkits.mplot3d import Axes3D
plot.rcParams['figure.figsize'] = (16, 9)
plot.style.use('ggplot')

# Leer Dataset
dataset = pd.read_csv('DatasetSGE_ValoresCorrectos.csv')

X = np.array(dataset[["05Adultos","06Dias_Reserva","14Edad_Actual_TITULAR"]])
print(X.shape)

# Nc = range(1, 20)
# kmeans = [KMeans(n_clusters=i) for i in Nc]
# score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
# plot.plot(Nc,score)
# plot.xlabel('Number of Clusters')
# plot.ylabel('Score')
# plot.title('Elbow Curve')
# plot.show()

kmeans = KMeans(n_clusters=4).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

# Predicting the clusters
labels = kmeans.predict(X)
# Getting the cluster centers
C = kmeans.cluster_centers_
colores = ['red', 'green', 'blue', 'cyan']
asignar = []
for row in labels:
    asignar.append(colores[row])

fig = plot.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=asignar, s=5)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c=colores, s=1000)


plot.show()

####Codigo que funciona 1

#Encontrar el numero optimo de clusters
# wcss = []
# for i in range (1,16):
#     kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 0)
#     kmeans.fit(X)
#     wcss.append(kmeans.inertia_)
# plot.plot(range(1,16),wcss)
# plot.title('Metodo de arco')
# plot.xlabel('Numero de clusters')
# plot.ylabel('wcss')
# plot.show()

#Algoritmo k-means y clustering

# kmeans = KMeans (n_clusters=3, init='k-means++', random_state= 0)
#
# y = kmeans.fit_predict(X)
# plot.scatter(X[y == 0,0], X[y== 0,1], s=25, c='red', label='Cluster 1')
# plot.scatter(X[y == 1,0], X[y== 1,1], s=25, c='blue', label='Cluster 2')
# plot.scatter(X[y == 2,0], X[y== 2,1], s=25, c='magenta', label='Cluster 3')
# # plot.scatter(X[y == 3,0], X[y== 3,1], s=25, c='black', label='Cluster 4')
# # plot.scatter(X[y == 4,0], X[y== 4,1], s=25, c='orange', label='Cluster 5')
# # plot.scatter(X[y == 5,0], X[y== 5,1], s=25, c='green', label='Cluster 6')
#
#
# plot.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=25, c='yellow', label='Centroid')
# plot.title('kmeans clustering')
# plot.xlabel('Dias reserva')
# plot.ylabel('Dias prereserva')
# plot.legend()
# plot.show()


####CODIGO QUE FUNCIONA 2
#Seleccionar Columnas Dataset
# #Dias reserva y prereserva
# X = dataset.iloc[:,[20,37]].values
#
# #Imprimir X
# # print (X)
#
# #Algoritmo k-means y clustering
#
# kmeans = KMeans(n_clusters=6)
# kmeans.fit(X)
#
# centroids = kmeans.cluster_centers_
# labels = kmeans.labels_
#
# print(centroids)
# print(labels)
#
# colors = ["g.", "r.", "b.", "c.", "y.", "orange"]
#
# for i in range(len(X)):
#     print("coordinate:", X[i], "label:", labels[i])
#     plot.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 1)
#
# plot.scatter(centroids[:,0], centroids[:,1], marker="x", s=150, linewidths=5, zorder = 10)
# plot.show()

#####PRUEBA 1
# import plotly.plotly as plot
# # import pandas as pd
#
# # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/alpha_shape.csv')
# dataset.head()
#
# scatter = dict(
#     mode = "markers",
#     name = "y",
#     type = "scatter3d",
#     x = dataset['Dias_Reserva'], y = dataset['Diferencia_Dias_Creacion_Prereserva_Reserva'], z = dataset['Edad_Actual_TITULAR'],
#     marker = dict( size=2, color="rgb(23, 190, 207)" )
# )
# clusters = dict(
#     alphahull = 7,
#     name = "y",
#     opacity = 0.1,
#     type = "mesh3d",
#     x = dataset['Dias_Reserva'], y = dataset['Diferencia_Dias_Creacion_Prereserva_Reserva'], z = dataset['Edad_Actual_TITULAR']
# )
# layout = dict(
#     title = '3d point clustering',
#     scene = dict(
#         xaxis = dict( zeroline=False ),
#         yaxis = dict( zeroline=False ),
#         zaxis = dict( zeroline=False ),
#     )
# )
# fig = dict( data=[scatter, clusters], layout=layout )
# # Use py.iplot() for IPython notebook
# plot.plot(fig, filename='mesh3d_sample')
########

# for line in dataset:
#     print (line[1])



# dataset.columns = [""]
# col_name = dataset.columns[0]
# dataset =dataset.rename(columns = {col_name:'DiasReserva'})

# DiasReserva = dataset.DiasReserva

# Diferencia_Dias_Creacion_Prereserva_Reserva = dataset.Diferencia_Dias_Creacion_Prereserva_Reserva


# print (dataset.columns)
# print (dataset.columns.DiasRerserva)
# print(dataset.head())