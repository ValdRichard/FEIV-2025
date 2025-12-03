import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

def leer_archivo_y_crear_dfs(filepath):
    df = pd.read_csv(filepath)

    columnas = df.columns.to_list()
    n = len(columnas) // 3

    dfs = {}

    for i in range(n):
        col_x  = columnas[3*i]
        col_ex = columnas[3*i + 1]
        col_y  = columnas[3*i + 2]

        nombre = col_x

        dfs[nombre] = df[[col_x, col_ex, col_y]].copy()

    return dfs

def ajuste_lineal_odr(x, ex, y):
    from scipy.odr import ODR, Model, RealData

    def f(B, x):
        return B[0] * x + B[1]
    modelo = Model(f)
    datos = RealData(x, y, sx=ex, sy=None)
    odr = ODR(datos, modelo, beta0=[1., 0.])
    salida = odr.run()
    m, b = salida.beta
    dm, db = salida.sd_beta

    return (m, b), (dm, db)

def graficar_ajustes(
    dfs,
    mostrar=True, 
    titulo="Ajustes lineales ODR",
    xlabel="X",
    ylabel="Y"
):
    
    plt.figure(figsize=(10, 6))

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

        (m, b), (dm, db) = ajuste_lineal_odr(x, ex, y)

        plt.errorbar(
            x, y, xerr=ex,
            fmt='o', capsize=3,
            color=color,
            label=f"Grupo {i+1}"
        )

        xline = np.linspace(x.min(), x.max(), 200)
        plt.plot(xline, m * xline + b, color="black", linewidth=1.4)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(titulo)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Mostrar o no el gráfico
    if mostrar:
        plt.show()

    # Si no lo muestro, devuelvo la figura para guardarla si querés
    return plt.gcf()

ruta = "./Experimento VI/Datos/"

dfs = leer_archivo_y_crear_dfs('Experimento VI/Datos/Presentacion tp6 - Calibracion.csv')
graficar_ajustes(dfs,
    mostrar=True,
    titulo="Rectas de calibración - Experimento VI",
    xlabel="Canales",
    ylabel="Energía (keV)"
)

