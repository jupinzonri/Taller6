"""
Taller_6_IA.py
Implementación de los algoritmos de Machine Learning (KNN, SVM y Árboles de Decisión) 
utilizando la librería Scikit-Learn sobre un dataset estándar de clasificación.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargar un dataset estándar para ilustrar los modelos (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Partición del Dataset (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =====================================================================
# ALGORITMO 1: K-Nearest Neighbors (KNN)
# =====================================================================
def ejecutar_knn():
    print("--- 1. Algoritmo K-Nearest Neighbors (KNN) ---")
    # Instanciar y entrenar el modelo KNN con k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Predicción y evaluación
    y_pred = knn.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    
    print(f"Precisión del modelo KNN: {precision:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# =====================================================================
# ALGORITMO 2: Support Vector Machine (SVM)
# =====================================================================
def ejecutar_svm():
    print("--- 2. Support Vector Machine (SVM) ---")
    # Instanciar y entrenar el modelo SVM con kernel RBF
    svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    
    # Predicción y evaluación
    y_pred = svm_model.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    
    print(f"Precisión del modelo SVM: {precision:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# =====================================================================
# ALGORITMO 3: Árboles de Decisión (Decision Trees)
# =====================================================================
def ejecutar_arbol_decision():
    print("--- 3. Árboles de Decisión ---")
    # Instanciar y entrenar el modelo de Árbol de Decisión
    tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
    tree_model.fit(X_train, y_train)
    
    # Predicción y evaluación
    y_pred = tree_model.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    
    print(f"Precisión del modelo de Árbol de Decisión: {precision:.4f}")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

if __name__ == "__main__":
    ejecutar_knn()
    print("\n" + "="*50 + "\n")
    ejecutar_svm()
    print("\n" + "="*50 + "\n")
    ejecutar_arbol_decision()
