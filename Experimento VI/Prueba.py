import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.special import erf

def graficar(x, y, xlabel, ylabel):
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.show()

def espectro(path, nombre):
    df = leer_spe(path, nombre)
    fondo = leer_spe(path, 'Fondo-18-9-1800s.Spe')
    df["Cuentas"] = df["Cuentas"] - fondo["Cuentas"]
    return df

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

ruta = "./Experimento VI/Datos/"

df_Am = espectro( ruta, 'Am241_4_11_2024.Spe') 