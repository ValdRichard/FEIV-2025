import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.special import erf

def leer_spe(path, nombre):
    with open(path + nombre, "r") as f:
        lines = [l.strip() for l in f]

    start = lines.index("$DATA:") + 10
    end = 763
    counts = [int(x) for x in lines[start:end] if x]

    df = pd.DataFrame({
        "Canal": range(len(counts)),
        "Cuentas": counts
    })
    return df

ruta = "./Experimento VI/Datos/"
df_Am = leer_spe(ruta, "Am241_4_11_2024.Spe")
from Analisis import m, sm, b, sb
df_Am["Energía"] = df_Am["Canal"] * m + b

x0_Am = df_Am["Canal"].values
x_Am = df_Am["Energía"].values
y_Am = df_Am["Cuentas"].values

x0_err_Am = np.full(len(x_Am), 1/2, dtype=float)
x_err_Am = np.sqrt( (m * x0_err_Am)**2 + (x0_Am * sm)**2 + sb**2 )
y_err_Am = np.sqrt(y_Am)
y_err_Am[y_err_Am == 0] = 0.0001


plt.figure(figsize=(8, 5))
plt.errorbar(
    x_Am, y_Am,
    xerr=x_err_Am, yerr=y_err_Am,
    fmt='o',                         # formato del punto (o = círculo)
    ecolor='gray',                   # color de las barras de error
    elinewidth=1,                    # grosor de línea de error
    capsize=3,                       # tamaño de los "topes" en las barras
    markersize=4,                    # tamaño de los puntos
    markeredgecolor='black',
    markerfacecolor='blue',
    alpha=0.8
    )

lineas_x = [3.25,3.66,11.87,13.95,16.79,17.75,20.78] # valores donde querés poner las líneas
lineas_labels = ['Ma','Mg','Ll','La1','Lb2','Lb1','Lg1']

for x, label in zip(lineas_x, lineas_labels):
    plt.axvline(x=x, linestyle='-', linewidth=2.4, color='red', alpha=0.7)
    plt.text(x, max(y_Am)*0.7, label, rotation=90,
             verticalalignment='top', horizontalalignment='right',
             color='red', fontsize=15, fontweight='bold')
    
plt.xlabel('Energía [keV]')
plt.ylabel('Cuentas')
plt.title('Espectro de Am-241')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()






archivos = ["Ag.spe","Co.spe","Cr.spe","Cu.spe","Fe.spe","Mn.spe","Mo.spe","Nb.spe","Pb.spe","Pd.spe","Ru.spe","Se.spe","Sn.spe","W.spe","Zn.spe","Zr.spe"]
titulos = ["Ag","Co","Cr","Cu","Fe","Mn","Mo","Nb","Pb","Pd","Ru","Se","Sn","W","Zn","Zr"]
for j,i in enumerate(archivos) : 
    df = leer_spe( ruta, i) 
    df["Energía"] = df["Canal"] * m + b
    x0 = df["Canal"].values
    x = df["Energía"].values
    y = df["Cuentas"].values

    x0_err = np.full(len(x), 1/2, dtype=float)
    x_err = np.sqrt( (m * x0_err)**2 + (x0 * sm)**2 + sb**2 )
    y_err = np.sqrt(y)
    y_err[y_err == 0] = 0.0001

    if j==2:

        # Eliminar los primeros 30 puntos
        x = x[30:]
        y = y[30:]
        x_err = x_err[30:]
        y_err = y_err[30:]


        plt.figure(figsize=(8, 5))
        plt.errorbar(
        x, y,
        xerr=x_err, yerr=y_err,
        fmt='o',                         # formato del punto (o = círculo)
        ecolor='gray',                   # color de las barras de error
        elinewidth=1,                    # grosor de línea de error
        capsize=3,                       # tamaño de los "topes" en las barras
        markersize=4,                    # tamaño de los puntos
        markeredgecolor='black',
        markerfacecolor='blue',
        alpha=0.8
        )

        lineas_x_Np = [13.73,13.96,16.01,16.80,17.74] # valores donde querés poner las líneas
        lineas_x_Ni = [7.40,8.19]
        lineas_x_Ag = [22.12,24.86,25.48]
        

        for x in lineas_x_Np:
            plt.axvline(x=x, linestyle='-', linewidth=2.4, color='green', alpha=0.7)
        for x in lineas_x_Ni:
            plt.axvline(x=x, linestyle='-', linewidth=2.4, color='red', alpha=0.7)
        for x in lineas_x_Ag:
            plt.axvline(x=x, linestyle='-', linewidth=2.4, color='gray', alpha=0.7)

        plt.axvline(x=13.36, linestyle='-', linewidth=2.4, color='black', alpha=0.7)
        plt.axvline(x=5.37, linestyle='-', linewidth=2.4, color='brown', alpha=0.7)
        plt.axvline(x=26.33, linestyle='-', linewidth=2.4, color='purple', alpha=0.7)
        plt.xlabel('Energía [keV]')
        plt.ylabel('Cuentas')
        plt.title('Espectro de emisión de RX del Cr')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


