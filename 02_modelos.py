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




