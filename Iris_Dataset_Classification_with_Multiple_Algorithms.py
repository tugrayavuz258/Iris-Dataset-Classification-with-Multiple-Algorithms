# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:56:56 2024

@author: TUGRA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns  # Seaborn daha estetik grafikler için

# Iris veri setini yükleyelim
iris = datasets.load_iris()

# Iris veri setini bir DataFrame'e dönüştürelim
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print(df)

# Hedef sınıfları ekleyelim
df['target_name'] = iris.target_names[iris.target]
print(df)

# X ve y değişkenlerini belirleyelim
x = df.iloc[:, 0:4].values
y = iris.target  # Y yanlış tanımlanmıştı, doğrudan target kullanılmalı

# Özellik isimleri
features = df.columns[:4]  # Bu satırdaki hata düzeltildi

# Her bir özellik için histogram çizdirme
for feature in features:
    plt.figure(figsize=(8,6))
    sns.histplot(data=df, x=feature, bins=20, kde=True)  # kde=True ile yoğunluk eğrisi eklenir
    plt.title(f"{feature} Dağılımı")
    plt.xlabel(feature)
    plt.ylabel("Frekans")
    plt.show()

# Eğitim ve test verilerini ayıralım
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Verileri standartlaştıralım
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# Logistic Regression modeli
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)
print("Logistic Regression Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# K-Nearest Neighbors modeli
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2, metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("K-Nearest Neighbors Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Support Vector Classifier modeli
from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Support Vector Classifier Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Gaussian Naive Bayes modeli
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Gaussian Naive Bayes Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Decision Tree Classifier modeli
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("Decision Tree Classifier Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Random Forest Classifier modeli
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("Random Forest Classifier Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)
