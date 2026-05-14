import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# cargar el dataset limpio
df = pd.read_csv("data/pagos_limpios.csv", encoding="utf-8")
df['FECHAPAGO'] = pd.to_datetime(df['FECHAPAGO'])

# configuracion de visual 
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 150


# colores a usar
AZUL = '#1a5276'
ROJO = '#c0392b'
VERDE = '#1e8449'
NARANJA = '#d35400'

print("Dataset cargado")
print(f"Registros cargados: {len(df):,}")