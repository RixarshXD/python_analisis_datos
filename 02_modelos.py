import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder


# se carga el dataset limpio para esta seccion
df = pd.read_csv("data/pagos_limpios.csv", encoding="utf-8")

print("Dataset cargado correctamente.")
print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns)}")

# verificamos si existe la variable objetivo
print(f"\nPagos ALTO (1): {df['PAGO_ALTO'].sum():,}")
print(f"Pagos BAJO (0): {(df['PAGO_ALTO'] == 0).sum():,}")

# preparacion de datos para el modelo

print("\n =========== Preparación de datos ===========")

# codificar las variables categoricas a numeros
le_region = LabelEncoder()
le_servicio = LabelEncoder()
le_asig = LabelEncoder()
le_item = LabelEncoder()

df['REG_COD'] = le_region.fit_transform(df['REGIONGEOGRAFICA'])
df['SER_COD'] = le_servicio.fit_transform(df['SERVICIO'])
df['ASIG_COD'] = le_asig.fit_transform(df['ASIGNACION'])
df['ITEM_COD'] = le_item.fit_transform(df['ITEM'])

print("Variables categoricas: ")
print(f"REGIONGEOGRAFICA -> REG_COD (0 a {df['REG_COD'].max()})")
print(f"SERVICIO -> SER_COD (0 a {df['SER_COD'].max()})")
print(f"ASIGNACION -> ASIG_COD (0 a {df['ASIG_COD'].max()})")
print(f"ITEM -> ITEM_COD (0 a {df['ITEM_COD'].max()})")

# definir las features (variables de entrada) y target (variable objetivo)

features = [
    'REG_COD', 'SER_COD', 'ASIG_COD', 'ITEM_COD',
    'MES_PAGO', 'AÑO_PAGO', 'TRIMESTRE'
]

x = df[features]
y = df['PAGO_ALTO']

print(f"\nVariables de entrada (X): {features}")
print("Variable objetivo (y): PAGO_ALTO")
print(f"Total muestro: {len(x):,}")

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nDivision train/test:")
print(f"Entrenamiento (80%): {len(x_train):,} registros")
print(f"Prueba (20%) {len(x_test):,} registros")



# arbol de decision
print("\n =========== Arbol de decision ===========")

# aqui se crea al modelo y se entrena
modelo = DecisionTreeClassifier(
    max_depth= 6, #nivel de profundidad
    min_samples_split= 50, #minimo de muestras para dividir un nodo
    min_samples_leaf= 20, #minimo de 20 registros 
    random_state= 42 # reproducibilidad del modelo
)

modelo.fit(x_train, y_train)
print("modelo entrenado.")

# predecir sobre el conjunto de prueba
y_pred = modelo.predict(x_test)
y_prob = modelo.predict_proba(x_test)[:, 1] #probabilidad de clase positiva

print("Predicciones realizadas sobre el conjunto de prueba.")

# metricas de evaluacion
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print("\n ========= MÉTRICAS DEL MODELO ========= \n")
print(f"  Accuracy:  {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Precision: {prec:.4f}  ({prec*100:.1f}%)")
print(f"  Recall:    {rec:.4f}  ({rec*100:.1f}%)")
print(f"  F1-Score:  {f1:.4f}  ({f1*100:.1f}%)")
print(f"  AUC-ROC:   {auc:.4f}  ({auc*100:.1f}%)")

print("\n ========= MATRIZ DE CONFUSIÓN ========= \n")
print(f"  Verdaderos Negativos (TN): {cm[0][0]:,}")
print(f"  Falsos Positivos     (FP): {cm[0][1]:,}")
print(f"  Falsos Negativos     (FN): {cm[1][0]:,}")
print(f"  Verdaderos Positivos (TP): {cm[1][1]:,}")

# Importancia de variables
print("\n ========= IMPORTANCIA DE VARIABLES ========= \n")
importancias = pd.Series(
    modelo.feature_importances_,
    index=features
).sort_values(ascending=False)

for var, imp in importancias.items():
    barra = "█" * int(imp * 50)
    print(f"  {var:<12}: {imp:.4f}  {barra}")