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
import seaborn as sns  # Seaborn ile grafikler daha estetik görünüyor

# Iris veri setini yükleyelim
iris = datasets.load_iris()

# Iris veri setini bir DataFrame'e dönüştürelim, bu bizim üzerinde çalışacağımız tablo
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Şimdilik verinin neye benzediğini görelim
print(df)

# Hedef sınıfları ekleyelim, yani hangi çiçek türü olduğu bilgisi
df['target_name'] = iris.target_names[iris.target]
print(df)  # Kontrol için tabloyu tekrar bastırıyoruz

# X, özellikleri (veri) ve y, hedef değişkeni (sınıf) olarak ayırıyoruz
x = df.iloc[:, 0:4].values  # Tüm özellikler
y = iris.target  # Hedef değişken (hangi çiçek türü olduğu)

# Her bir özellik için histogram çizdiriyoruz, dağılımı görmek için faydalı
features = df.columns[:4]  # İlk dört sütun, çiçek özellikleri
for feature in features:
    plt.figure(figsize=(8,6))
    sns.histplot(data=df, x=feature, bins=20, kde=True)  # kde=True: yoğunluk eğrisi ekliyor
    plt.title(f"{feature} Dağılımı")
    plt.xlabel(feature)
    plt.ylabel("Frekans")
    plt.show()  # Grafikler arka arkaya gösterilecek

# Şimdi veriyi eğitim ve test kümelerine ayırıyoruz (eğitim %80, test %20)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

# Verileri standartlaştırıyoruz, böylece her özellik aynı ölçeğe sahip olur
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)  # Eğitim verisini standartlaştırıyoruz
X_test = sc.transform(x_test)  # Test verisini de aynı şekilde dönüştürüyoruz

# Logistic Regression ile modelimizi oluşturuyoruz
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)  # Modeli eğitim verisiyle eğitiyoruz
y_pred = logr.predict(X_test)  # Test verisini kullanarak tahmin yapıyoruz
print("Logistic Regression Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)  # Confusion matrix'i hesaplıyoruz
print(cm)

# K-Nearest Neighbors (KNN) ile model oluşturma
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2, metric='minkowski')
knn.fit(X_train, y_train)  # KNN modeli eğitiliyor
y_pred = knn.predict(X_test)  # Test verisinde tahmin yapıyoruz
print("K-Nearest Neighbors Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Support Vector Classifier (SVC) modeli
from sklearn.svm import SVC
svc = SVC(kernel='rbf')  # Radial basis function (RBF) çekirdeği kullanılıyor
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Support Vector Classifier Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Gaussian Naive Bayes modeli
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)  # Naive Bayes modelini eğitiyoruz
y_pred = gnb.predict(X_test)  # Tahmin yapılıyor
print("Gaussian Naive Bayes Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Decision Tree Classifier modeli
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')  # Entropy kullanarak karar ağacı oluşturuyoruz
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print("Decision Tree Classifier Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Random Forest Classifier modeli (10 ağaçlı orman)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy')
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print("Random Forest Classifier Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
print(cm)
