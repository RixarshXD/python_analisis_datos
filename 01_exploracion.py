# importar librerias
import pandas as pd

# ======
# DATASET 01
# ======
# cargar el dataset
df = pd.read_csv("data/pagos.csv", encoding="utf-8")

# verificamos el data set
print("\n =========FILAS Y COLUMNAS========= \n")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

# visualizar las 3 primeras filas
print("\n =========PRIMERAS FILAS========= \n")
print(df.head(3))

# nombre de las columnas
print("\n =========NOMBRE DE LAS COLUMNAS========= \n")
print("\n".join(df.columns.to_list()))

# tipos de datos
print("\n =========TIPOS DE DATOS========= \n")
print(df.dtypes.to_string())

# estadisticas descriptivas
print("\n =========ESTADISTICAS DESCRIPTIVAS========= \n")
print(df.describe())


# ======
# ESTADISTICAS 02
# ======
# ahora hacemos las estadisticas
print("\n =========ESTADISTICAS========= \n")

# configuracion para formatos de numero
pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 120)

# aqui muestro la columna que es relevante que seria TOTALPAGADO
print("\n ========= TOTALPAGADO ========= \n")
stats = df["TOTALPAGADO"].describe()
labels = {
    "count": "Total registros",
    "mean": "Promedio",
    "std": "Desv_Estandar",
    "min": "Minimo",
    "25%": "Percentil del 25%",
    "50%": "Mediana",
    "75%": "Percentil del 75%",
    "max": "Maximo",
}
for key, label in labels.items():
    print(f"{label}: ${stats[key]:>20,.0f} CLP")

# verificar cuantos valores negativos tenemos en el dataset
negativos = (df["TOTALPAGADO"] < 0).sum()
ceros = (df["TOTALPAGADO"] == 0).sum()

# mostrar los resultados
print("\n ========= VALORES NEGATIVOS Y CEROS ========= \n")
print(f"Valores negativos: {negativos}")
print(f"Ceros: {ceros}")

# valores nulos
print("\n ========= VALORES NULOS ========= \n")
nulos = df.isnull().sum()
pct = (nulos / len(df) * 100).round(2)

# mostramos un resumen
# el 61% de los pagos es porque no pasaron por el proceso administrativo
# simplemente fueron pagados directos sin ese tramite intermedio.
resumen_nulos = pd.DataFrame({"Nulos": nulos, "Porcentaje %": pct})
resumen_nulos = resumen_nulos[resumen_nulos["Nulos"] > 0]
# luego se imprime en pantalla
print(resumen_nulos.to_string())

# mostrar los valores unicos
print("\n ========= VALORES UNICOS ========= \n")
for col in ["REGIONGEOGRAFICA", "SERVICIO", "ITEM", "ASIGNACION"]:
    print(f" {col}: {df[col].nunique()} valores unicos")

print("\n ========= REGIONES DISPONIBLES =========")
for r in sorted(df["REGIONGEOGRAFICA"].unique()):
    print(f" - {r}")


# ======
# LIMPIEZA 03
# ======
print("\n ========= LIMPIEZA DEL DATASET ========= \n")
# tecnica para cargar el dataset en memoria
df = pd.read_csv("data/pagos.csv", encoding="utf-8", low_memory=False)

# convertir la columna de fechas
df["FECHAPAGO"] = pd.to_datetime(df["FECHAPAGO"], format="%d/%m/%y", errors="coerce")
df["FECHA_DOCUMENTO"] = pd.to_datetime(
    df["FECHA_DOCUMENTO"], format="%d/%m/%y", errors="coerce"
)
df["FECHA_INGRESO_MOP"] = pd.to_datetime(
    df["FECHA_INGRESO_MOP"], format="%d/%m/%y", errors="coerce"
)

print("Fechas convertidas")
print(f"Tipo de FECHAPAGO ahora: {df['FECHAPAGO'].dtype}")

# convertir el rut a texto
df["RUTADJUDICADO"] = df["RUTADJUDICADO"].astype(str)
print("RUTADJUDICADO convertido a texto")

# eliminar filas con montos invalidos
antes = len(df)
df = df[df["TOTALPAGADO"] > 0]
eliminados_monto = antes - len(df)
print(f"Filas eliminadas por montos invalidos: {eliminados_monto}")

# eliminar filas sin nombre de proveedor
print("Filas eliminadas por nombre de proveedor: ")
antes = len(df)
df = df[df["NOMBREADJUDICADO"].notna()]
eliminados_nombre = antes - len(df)
print(f"{eliminados_nombre}")


# verificar que ya no queden nulos en las columnas
print("\n ========= VERIFICACION DE VALORES NULOS ========= \n")
criticas = [
    "FECHAPAGO",
    "TOTALPAGADO",
    "REGIONGEOGRAFICA",
    "SERVICIO",
    "NOMBREADJUDICADO",
]
for col in criticas:
    nulos = df[col].isnull().sum()
    estado = "OK" if nulos == 0 else f" {nulos} nulos"
    print(f" {col}: {estado}")

print(f"\nRegistros originales: 119,913")
print(f"Registros finales: {len(df):,}")
print(f"Registros eliminados: {119913 - len(df):,}")


# creando variables nuevas antes de generar el nuevo archivo xlsx
print("\n ========= VARIABLES NUEVAS ========= \n")

# variables de fechas de pago
df["MES_PAGO"] = df["FECHAPAGO"].dt.month
df['AÑO_PAGO'] = df['FECHAPAGO'].dt.year
df['TRIMESTRE'] = df['FECHAPAGO'].dt.quarter

print("Variables creadas: MES_PAGO, AÑO_PAGO, TRIMESTRE")

# dias de tramitacion
df['DIAS_TRAMITACION'] = (df['FECHAPAGO'] - df['FECHA_INGRESO_MOP']).dt.days
disponibles = df['DIAS_TRAMITACION'].notna().sum()
print(f"Variable creada: DIAS_TRAMITACION")
print(f"Disponible en {disponibles:,} registros ({disponibles/len(df)*100:.1f}%)")


# variable binaria para el modelo
mediana = df['TOTALPAGADO'].median()
df['PAGO_ALTO'] = (df['TOTALPAGADO'] > mediana).astype(int)

pagos_altos = df['PAGO_ALTO'].sum()
pagos_bajos = (df['PAGO_ALTO'] == 0).sum()
print(f"Variable creada: PAGO_ALTO (1 = monto < mediana)")
print(f"Mediana usada. ${mediana:,.0f} CLP")
print(f"Pagos ALTO (1): {pagos_altos:,}")
print(f"Pagos BAJO (0): {pagos_bajos:,}")

# verificar el dataset final 
print("\n ========= DATASET FINAL ========= \n")
print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns)}")
print(f"\n Columnas discponibles:")
for col in df.columns:
    print(f" - {col}: {df[col].dtype}")

# guardar el dataset limpio
df.to_csv("data/pagos_limpios.csv", index=False, encoding="utf-8")
print("\nDataset limpio guardado en: 'data/pagos_limpios.csv'")