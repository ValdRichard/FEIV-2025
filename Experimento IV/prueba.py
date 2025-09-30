import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def leer_spe(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f]

    start = lines.index("$DATA:") + 10
    end = lines.index("$ROI:")
    counts = [int(x) for x in lines[start:end] if x]

    df = pd.DataFrame({
        "Canal": range(len(counts)),
        "Cuentas": counts
    })
    return df
def calibrar_picos(df, energias_referencia):
    # Detectar todos los picos
    peaks, properties = find_peaks(df["Cuentas"], height=max(df["Cuentas"])*0.05, distance=20)
    
    # Obtener el canal central de cada pico (el máximo dentro de cada pico detectado)
    canales_picos = []
    for p in peaks:
        # max del pico en su vecindad (ej. ±10 canales)
        left = max(p-10, 0)
        right = min(p+10, len(df)-1)
        local_max = df["Cuentas"].iloc[left:right+1].idxmax()
        canales_picos.append(local_max)

    canales_picos = np.array(sorted(canales_picos))  # ordenar por canal
    print("Picos detectados (canal central):", canales_picos)
    
    # Seleccionamos los dos picos más cercanos a donde esperamos los picos de referencia
    # Por ejemplo, asumimos que el primero es el de rayos X (bajo canal) y el segundo el fotopico
    canales_seleccionados = [canales_picos[0], canales_picos[-1]]

    # Ajuste lineal
    a, b = np.polyfit(canales_seleccionados, energias_referencia, 1)
    df["Energia_keV"] = a * df["Canal"] + b

    print(f"Calibración: E = {a:.4f} * canal + {b:.4f}")
    print(f"Picos usados para calibrar: {canales_seleccionados}, energías: {energias_referencia}")
    
    return df, (a,b)


def graficar(df):
    plt.figure(figsize=(8,5))
    plt.scatter(df["Energia_keV"], df["Cuentas"])
    plt.xlabel("Energía [keV]")
    plt.ylabel("Cuentas")
    plt.title("Espectro calibrado")
    plt.grid(alpha=0.3)
    plt.show()

# --- Ejemplo de uso ---
ruta = "./Experimento IV/Datos/Cs137-cu.Spe"
df = leer_spe(ruta)

# Energías conocidas de referencia (rayos X Cs ~32 keV, fotopico Cs-137 ~662 keV)
energias_referencia = [32, 662]

df_cal, calib = calibrar_picos(df, energias_referencia)
graficar(df_cal)
