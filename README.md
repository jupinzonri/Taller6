# Taller 6: Introducción a Machine Learning
**Nombre:** Juan Felipe Pinzón Rincón

## Análisis de los Algoritmos Seleccionados

### 1. Algoritmo K-Nearest Neighbors (KNN)
* Es un algoritmo de aprendizaje supervisado utilizado tanto para clasificación como para regresión. Funciona encontrando los $k$ ejemplos más cercanos en el espacio de características a una nueva instancia mediante métricas de distancia (como la distancia euclidiana).
* No requiere una fase de entrenamiento compleja (es un algoritmo "perezoso" o *lazy learning*), ya que memoriza el conjunto de datos de entrenamiento y realiza la predicción basada en la votación mayoritaria de sus vecinos más cercanos.

### 2. Support Vector Machine (SVM)
* Es un potente algoritmo de aprendizaje supervisado enfocado en la clasificación y regresión. Su objetivo principal es encontrar un hiperplano óptimo en un espacio de características $n$-dimensional que separe de forma óptima las clases maximizando el margen entre ellas.
* Utiliza funciones de kernel para transformar datos no linealmente separables a espacios de mayor dimensión, lo que lo hace sumamente efectivo en problemas complejos de alta dimensionalidad.

### 3. Árboles de Decisión (Decision Trees)
* Es un modelo predictivo supervisado que estructura las decisiones y sus posibles consecuencias en forma de diagrama de flujo jerárquico, compuesto por nodos de decisión y nodos terminales (hojas).
* Cada nodo interno evalúa una característica específica mediante umbrales (como el índice Gini o la ganancia de información), dividiendo los datos en subconjuntos cada vez más homogéneos hasta llegar a una clasificación o valor de predicción.
