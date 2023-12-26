# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:16:14 2023

@author: ivoto
"""

# Natural Language Processing 

#Importamos librerias 
import numpy as np 
import matplotlib as plt
import pandas as pd 

#Importamos el dataSet 

dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)

#Limpieza de texto 

import re
import nltk #Natural Language toolkit
nltk.download("stopwords") #Descarga las palabras innecesarias
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #Eliminamos todo menos las letras, y lo remplazamos por un espacio
    review = review.lower() #Pasamos todas las letras a minuscula
    review = review.split()
    ps = PorterStemmer() #Clase para convertir palabras a su forma infinitiva(stemm es eliminar la raiz, convertirlo en infinitivo)
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review = '  '.join(review)
    corpus.append(review)

# Crear el Bag of Words (Proceso de  tokenizacion)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #Le doy una frase y me la transforma a vector #max_feactures busca las frecuencia y las mas relevante (Ej nombres no hace faltan que vayan)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


#Training & Test 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0 )

'''
#Ajustar el clasificador
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#Prediccion de los resultados con el conjunto de test 
y_pred = classifier.predict(X_test)

'''

#Ajustar el clasificador
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy")
classifier.fit(X_train, y_train )

#Prediccion de los resultados con el conjunto de test 
y_pred = classifier.predict(X_test)

#Elaboramos la matriz de confusion 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



 














