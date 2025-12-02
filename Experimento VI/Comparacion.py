import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# -------------------------------------------------------------
# 1) Leer archivo y crear los 6 dataframes con 3 arrays cada uno
# -------------------------------------------------------------
def leer_archivo_y_crear_dfs(filepath):
    df = pd.read_csv(filepath)

    columnas = df.columns.to_list()

    # Cantidad de tríos (x, ex, y)
    n = len(columnas) // 3

    dfs = {}

    for i in range(n):
        col_x  = columnas[3*i]
        col_ex = columnas[3*i + 1]
        col_y  = columnas[3*i + 2]

        nombre = col_x  # nombre del dataset = nombre de la primera columna

        dfs[nombre] = df[[col_x, col_ex, col_y]].copy()

    return dfs


# -------------------------------------------------------------
# 2) Ajuste lineal por ODR
# -------------------------------------------------------------
def ajuste_lineal_odr(x, ex, y):
    from scipy.odr import ODR, Model, RealData

    # Modelo lineal f(p, x) = p[0]*x + p[1]
    def f(B, x):
        return B[0] * x + B[1]

    modelo = Model(f)
    datos = RealData(x, y, sx=ex, sy=None)
    odr = ODR(datos, modelo, beta0=[1., 0.])

    salida = odr.run()

    # parámetros
    m, b = salida.beta

    # errores de parámetros
    dm, db = salida.sd_beta

    return (m, b), (dm, db)



# -------------------------------------------------------------
# 3) Graficar los 6 ajustes en un solo gráfico
# -------------------------------------------------------------
def graficar_ajustes(dfs):
    plt.figure(figsize=(10, 6))

    # Elegí aquí los colores para los puntos
    colores = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown"
    ]

    for i, (nombre, data) in enumerate(dfs.items()):
        x  = data.iloc[:, 0].to_numpy()
        ex = data.iloc[:, 1].to_numpy()
        y  = data.iloc[:, 2].to_numpy()

        color = colores[i % len(colores)]

        # Ajuste ODR
        (m, b), (dm, db) = ajuste_lineal_odr(x, ex, y)

        # Puntos (color elegido)
        plt.errorbar(
            x, y, xerr=ex,
            fmt='o', capsize=3,
            color=color, label=f"{nombre}"
        )

        # Recta (NEGRA siempre)
        xline = np.linspace(x.min(), x.max(), 200)
        plt.plot(xline, m * xline + b, color="black", linewidth=1.4)
    
    plt.xlabel("Cuentas")
    plt.ylabel("Energía [keV]")
    plt.title("Rectas de calibración - Experimento VI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# -------------------------------------------------------------
# Ejemplo de uso:
# -------------------------------------------------------------
dfs = leer_archivo_y_crear_dfs("C:/Users/Usuario/Desktop/FEIV-2025/FEIV-2025/Experimento VI/Datos/Presentacion tp6 - Calibracion.csv")
graficar_ajustes(dfs)
