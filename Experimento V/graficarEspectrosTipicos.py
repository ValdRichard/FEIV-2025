import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.special import erf
from scipy.optimize import curve_fit


# === Función base ===
def graficar(x, y, xlabel, ylabel):
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, s=5, marker='.', color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.show()

def cortar_datos(izquierda, derecha, x, y, x_err, y_err):
    x_data = x[izquierda:derecha]
    y_data = y[izquierda:derecha]
    x_err = x_err[izquierda:derecha]
    y_err = y_err[izquierda:derecha]
    return x_data, y_data, x_err, y_err

# === Función con errores ===
def graficar_con_error(x, y, xerr, yerr, xlabel, ylabel, titulo=None):
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt='o',
        ecolor='gray',
        elinewidth=1,
        capsize=3,
        markersize=3,              # más chico
        markeredgecolor='black',
        markerfacecolor='blue',
        alpha=0.8
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if titulo:
        plt.title(titulo)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# === NUEVA FUNCIÓN: scatter + rangos + líneas ===
def graficar_con_rango(x, y, xlabel, ylabel, rangos=None, lineas=None, titulo=None):
    """
    Grafica el espectro con puntos (scatter) y permite resaltar regiones o líneas.

    rangos: lista de dicts {"xmin", "xmax", "label", "color"}
    lineas: lista de dicts {"x", "label", "color"}
    """
    plt.figure(figsize=(9, 5))
    
    # Gráfico base con puntos pequeños
    plt.scatter(x, y, s=9, marker='8', color='blue', alpha=1, label="Datos")
    
    # Sombrear rangos (fotopicos)
    if rangos:
        for r in rangos:
            xmin, xmax = r.get("xmin"), r.get("xmax")
            color = r.get("color", "orange")
            label = r.get("label", None)
            plt.axvspan(xmin, xmax, color=color, alpha=0.25, label=label)
    
    # Líneas verticales
    if lineas:
        for l in lineas:
            x_line = l.get("x")
            color = l.get("color", "red")
            label = l.get("label", None)
            plt.axvline(x=x_line, color=color, linestyle="--", lw=1.2, alpha=0.8, label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if titulo:
        plt.title(titulo)
    
    # Mostrar leyenda solo si hay etiquetas
    if (rangos or lineas) and any(
        (r.get("label") for r in (rangos or []))
        or (l.get("label") for l in (lineas or []))
    ):
        plt.legend()
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# === Lectura de archivo SPE ===
def leer_spe(path, nombre):
    with open(path + nombre, "r") as f:
        lines = [l.strip() for l in f]

    start = lines.index("$DATA:") + 10
    end = lines.index("$ROI:")
    counts = [int(x) for x in lines[start:end] if x]

    df = pd.DataFrame({
        "Canal": range(len(counts)),
        "Cuentas": counts
    })
    return df


# === Espectro con resta de fondo ===
def espectro(path, nombre):
    df = leer_spe(path, nombre)
    fondo = leer_spe(path, 'Fondo-18-9-1800s.Spe')
    df["Cuentas"] = df["Cuentas"] - fondo["Cuentas"]
    return df


# === Ejemplo de uso ===
ruta = "./Experimento V/Datos/"

df_Co_malos = espectro(ruta, 'Co60-18-9.Spe')
graficar_con_rango(
    df_Co_malos["Canal"], df_Co_malos["Cuentas"],
    "Canal", "Cuentas",
    rangos=[
        {"xmin": 640, "xmax": 740, "label": "Fotopico 1173.2 keV", "color": "red"},
        {"xmin": 750, "xmax": 820, "label": "Fotopico 1332.5 keV", "color": "green"}
    ],
    titulo="Espectro de $^{60}$Co con fotopicos marcados"
)


# === Ejemplo de uso ===
ruta = "./Experimento V/Datos/"

df_Co_malos = espectro(ruta, 'Na22-18-9-1800s.Spe')
graficar_con_rango(
    df_Co_malos["Canal"], df_Co_malos["Cuentas"],
    "Canal", "Cuentas",
    rangos=[
        {"xmin": 250, "xmax": 360, "label": "Fotopico 511 keV", "color": "red"},
        {"xmin": 700, "xmax": 800, "label": "Fotopico 1274.5 keV", "color": "green"}
    ],
    titulo="Espectro de $^{22}$Na con fotopicos marcados"
)


# # === Ejemplo de uso ===
# ruta = "./Experimento V/Datos/"

# df_Co_malos = espectro(ruta, 'Cs137-18-9-1800s.Spe')
# graficar_con_rango(
#     df_Co_malos["Canal"], df_Co_malos["Cuentas"],
#     "Canal", "Cuentas",
#     rangos=[
#         {"xmin": 1, "xmax": 40, "label": "Fotopico 1173.2 keV", "color": "red"},
#         {"xmin": 350, "xmax": 460, "label": "Fotopico 1332.5 keV", "color": "green"}
#     ],
#     titulo="Espectro de Co-60 con fotopicos marcados"
# )

