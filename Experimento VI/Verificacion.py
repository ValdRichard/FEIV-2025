import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.special import erf


def fit_lineal(x, y, err_x=None, err_y=None, mostrar_grafica=True, label_x = 'Eje X', label_y = 'Eje Y', titulo = 'Title'):
    # Conversión a arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Manejo de errores
    if err_x is None:
        err_x = np.full_like(x, 1e-6, dtype=float)
    elif np.isscalar(err_x):
        err_x = np.full_like(x, err_x, dtype=float)

    if err_y is None:
        err_y = np.full_like(y, 1e-6, dtype=float)
    elif np.isscalar(err_y):
        err_y = np.full_like(y, err_y, dtype=float)


    # Modelo lineal
    def f_lineal(beta, x):
        m, b = beta
        return m * x + b

    modelo = Model(f_lineal)
    data = RealData(x, y, sx=err_x, sy=err_y)
    betai = [2.0, 0.0]

    odr = ODR(data, modelo, beta0=betai)
    out = odr.run()

    # Parámetros ajustados
    m, b = out.beta
    sm, sb = out.sd_beta

    # Predicción del modelo
    y_pred = f_lineal(out.beta, x)

    # Coeficiente de determinación R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    if mostrar_grafica:  

        plt.figure(figsize=(10,6))
        plt.errorbar(x, y, xerr=err_x, yerr=err_y, 
                     fmt='o', alpha=0.5, label='Datos', 
                     color='orange', capsize=3)
        
        x_fit = np.linspace(np.min(x), np.max(x), 1000)
        y_fit = f_lineal(out.beta, x_fit)
        
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                 label=(f'Ajuste ODR\n'
                        f'm={m:.2f}±{sm:.2f}\n'
                        f'b={b:.2f}±{sb:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(titulo)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return m, sm, b, sb, r2

ruta = "./Experimento VI/Datos/"


def leer_datos_espectros(path_csv):
    
    df_raw = pd.read_csv(path_csv, header=None, dtype=str)
    colnames = df_raw.iloc[0].tolist()
    df = df_raw.iloc[1:].copy()
    df.columns = colnames
    df = df.dropna(axis=1, how='all')
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.reset_index(drop=True)

    return df

def convertir_energias_a_eV(df):
    series = ["Ka", "Kb", "La", "Lb", "Lg", "Ll"]
    
    for serie in series:
        df[serie] = df[serie] * 1000          # Energía → eV
        df[f"err{serie}"] = df[f"err{serie}"] * 1000  # Error → eV

    return df

def formato_fisica(valor, error):
    """
    Devuelve una cadena del tipo "valor(error_en_ultimos_digitos)".
    Ejemplo: 0.3037 ± 0.0006 -> "0.3037(6)"
    """
    # Casos especiales
    if valor is None or error is None:
        return "nan"
    try:
        valor = float(valor)
        error = float(error)
    except Exception:
        return "nan"
    if np.isnan(valor) or np.isnan(error):
        return "nan"
    if error == 0:
        # mostrar sin paréntesis si no hay error (o podrías mostrar (0))
        # aquí devolvemos valor sin paréntesis
        # formateamos con máxima precisión razonable
        return f"{valor:.6g}"

    # orden de magnitud del error
    emin = int(np.floor(np.log10(abs(error))))
    # número de decimales que necesitamos mostrar:
    # si emin < 0 -> decimales = -emin
    # si emin >=0 -> decimales = 0 (redondeamos a unidades, decimales 0)
    decimales = max(-emin, 0)

    # calculamos el entero de la incertidumbre en unidades del último dígito mostrado
    factor = 10**decimales
    err_entero = int(round(error * factor))

    # si err_entero es 0 por redondeo (ocurre con errores muy pequeños), aumentamos decimales
    # hasta que err_entero != 0 o hasta un tope razonable (10 decimales)
    tope = 10
    while err_entero == 0 and decimales < tope:
        decimales += 1
        factor = 10**decimales
        err_entero = int(round(error * factor))

    # redondear el valor a los decimales calculados
    valor_red = round(valor, decimales)

    # formatear valor con los decimales exactos (para que las posiciones coincidan)
    if decimales > 0:
        fmt_val = f"{valor_red:.{decimales}f}"
    else:
        fmt_val = f"{int(round(valor_red))}"

    return f"{fmt_val}({err_entero})"

def lineal(B, x):
    m, b = B
    return m * x + b

def graficar_series_odr(df):
    
    series = ["Ka", "Kb", "La", "Lb", "Lg", "Ll"]

    for serie in series:
        # Extraer columnas dinámicamente
        Z = df[f"Z_{serie}"]
        errZ = df[f"errZ_{serie}"]
        E = df[serie]
        errE = df[f"err{serie}"]

        # Filtrar filas válidas
        mask = ~np.isnan(Z) & ~np.isnan(E)
        Z = Z[mask]
        errZ = errZ[mask]
        E = E[mask]
        errE = errE[mask]

        # Transformación: sqrt(E)
        x = np.sqrt(E)
        err_x = errE / (2 * np.sqrt(E))

        # Preparar datos para ODR
        data = RealData(x, Z, sx=err_x, sy=errZ)
        model = Model(lineal)
        odr = ODR(data, model, beta0=[1, 0])  # valores iniciales
        out = odr.run()

        m, b = out.beta
        m_err, b_err = out.sd_beta

        # ---- G R Á F I C O ----
        plt.figure(figsize=(6,4))

        # Datos experimentales con errores
        plt.errorbar(
            x, Z, xerr=err_x, yerr=errZ,
            fmt='o', capsize=3, label="Datos"
        )

        # Línea ajustada
        x_fit = np.linspace(min(x), max(x), 200)
        plt.plot(x_fit, lineal(out.beta, x_fit), label=f"Ajuste ODR\nm={m:.3f}±{m_err:.3f}\nb={b:.3f}±{b_err:.3f}")

        plt.xlabel(r"$\sqrt{E}$ [√keV]")
        plt.ylabel("Z")
        plt.title(f"Serie {serie}: ODR de Z vs √E")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

df_datos = leer_datos_espectros('Experimento VI/Datos/Datos tpVI - Hoja 3.csv')
df_datos_eV = convertir_energias_a_eV(df_datos)
#graficar_series_odr(df_datos_eV)

def graficar_todas_las_series_juntas(df):
    m_dict = {}
    m_err_dict = {}

    # Nombres de las series
    series = ["Ka", "Kb", "La", "Lb", "Lg", "Ll"]
    etiquetas = {
        "Ka": "Kα",
        "Kb": "Kβ",
        "La": "Lα",
        "Lb": "Lβ",
        "Lg": "Lγ",
        "Ll": "Lℓ"
    }

    colores = {
        "Ka": "purple",
        "Kb": "purple",
        "La": "red",
        "Lb": "red",
        "Lg": "red",
        "Ll": "red"
    }

    color_lineas = {
        "Ka": "green",
        "Kb": "green",
        "La": "blue",
        "Lb": "blue",
        "Lg": "blue",
        "Ll": "blue"      # Ejemplo
    }

    plt.figure(figsize=(8,6))
    
    for serie in series:

        Z = df[f"Z_{serie}"]
        errZ = df[f"errZ_{serie}"]
        E = df[serie]
        errE = df[f"err{serie}"]

        # Filtrar datos válidos
        mask = ~np.isnan(Z) & ~np.isnan(E)
        Z = Z[mask]
        errZ = errZ[mask]
        E = E[mask]
        errE = errE[mask]

        # Transformación sqrt(E)
        x = np.sqrt(E)
        err_x = errE / (2 * np.sqrt(E))

        # Preparar ODR
        data = RealData(x, Z, sx=err_x, sy=errZ)
        model = Model(lineal)
        odr = ODR(data, model, beta0=[1, 0])
        out = odr.run()

        m, b = out.beta
        m_err, b_err = out.sd_beta

        m_dict[serie] = m
        m_err_dict[serie] = m_err

        m_fmt = formato_fisica(m, m_err)
        b_fmt = formato_fisica(b, b_err)

        # Dibujar recta ajustada
        x_fit = np.linspace(min(x), max(x), 200)
        y_fit = lineal(out.beta, x_fit)

        plt.plot(
        x_fit, y_fit,
        color=color_lineas[serie],
        linewidth=2,
        label=f"{etiquetas[serie]}:  m={m_fmt}, b={b_fmt}"
        )

        # Dibujar puntos experimentales
        plt.errorbar(x, Z, xerr=err_x, yerr=errZ,
                     fmt='o', capsize=3,
            color=colores[serie], alpha=0.7)

    plt.xlabel(r"$\sqrt{E}$ [√eV]", fontsize=20)
    plt.ylabel("Número atómico Z", fontsize=20)
    plt.title("Verificación de la Ley de Moseley", fontsize=20)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return m_dict, m_err_dict

m_dict, m_err_dict = graficar_todas_las_series_juntas(df_datos_eV)

h = 4.135667696e-15   # eV·s
c = 299792458         # m/s

series = ["Ka", "Kb", "La", "Lb", "Lg", "Ll"]

delta = {
    "Ka": 3/4,
    "Kb": 8/9,
    "La": 5/36,
    "Lb": 3/16,
    "Lg": 21/100,
    "Ll": 5/36
}

R_dict = {}
R_err_dict = {}

for serie in series:
    m = m_dict[serie]
    errm = m_err_dict[serie]

    R = 1 / (h * c * delta[serie] * m**2)
    errR = (2 * errm) / (h * c * delta[serie] * m**3)

    R_dict[serie] = R
    R_err_dict[serie] = errR

# Mostrar resultados
for serie in series:
    R = R_dict[serie]
    errR = R_err_dict[serie]
    
    R_fmt = formato_fisica(R, errR)
    print(f"{serie}: R = {R_fmt}")
