import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
def fit_lineal(Vs, f, scale=1e14):
    """
    Ajuste lineal ODR con errores en V (3% + 0.01) y en f (típico 1e8),
    devuelve m, sm, b, sb, chi2 reducido y R2
    """
    Vs = np.array(Vs, dtype=float)
    f = np.array(f, dtype=float)

    # Errores
    errV = 0.03 * Vs + 0.01
    errF = np.full_like(f, 1e8, dtype=float)

    # Escalado
    f_scaled = f / scale
    errF_scaled = errF / scale

    # Modelo lineal
    def f_lineal(beta, x):
        m, b = beta
        return m * x + b

    modelo = Model(f_lineal)
    data = RealData(f_scaled, Vs, sx=errF_scaled, sy=errV)
    beta0 = [4.14e-15 * scale, 0.0]

    odr = ODR(data, modelo, beta0=beta0)
    out = odr.run()

    m_scaled, b = out.beta
    sm_scaled, sb = out.sd_beta

    m, sm = m_scaled / scale, sm_scaled / scale

    # Chi² reducido
    chi2_red = out.res_var  # varianza residual reducida

    # R²
    y_mean = np.mean(Vs)
    ss_tot = np.sum((Vs - y_mean) ** 2)
    ss_res = np.sum(out.delta ** 2)
    r2 = 1 - ss_res / ss_tot

    return m, sm, b, sb, chi2_red, r2


def analisis_tres_series(f, sin_filtro, con_filtro, pasco, scale=1e14, label="", nombres_colores=None):
    datasets = {
        "Datos - sin filtro": sin_filtro,
        "Datos - con filtro": con_filtro,
        "Datos - con filtro PASCO": pasco
    }

    colores = ["tab:blue", "tab:orange", "tab:green"]
    marcadores = ["D", "s", "v"]       # círculo, cuadrado, diamante
    tamanios = [6, 6, 6]               # tamaños ligeramente distintos

    plt.figure(figsize=(7, 6))

    resultados = {}
    funcion_label_agregada = False

    for (nombre, Vs), color, marker, msize in zip(datasets.items(), colores, marcadores, tamanios):
        m, sm, b, sb, chi2_red, r2 = fit_lineal(Vs, f, scale=scale)

        resultados[nombre] = {
            "m": m, "sm": sm,
            "b": b, "sb": sb,
            "chi2_red": chi2_red,
            "R2": r2
        }

        # Dibujar puntos
        errV = 0.03 * np.array(Vs) + 0.01
        errF = np.full_like(f, 1e8)
        plt.errorbar(f, Vs, yerr=errV, xerr=errF,
                     fmt=marker, markersize=msize,
                     ecolor="gray", capsize=3,
                     label=f"{nombre}", alpha=0.8, color=color)

        # Dibujar ajuste
        f_fit = np.linspace(min(f), max(f), 200)
        Vs_fit = m * f_fit + b

        if not funcion_label_agregada:
            label_ajuste = f"y = m·f + b"
            funcion_label_agregada = True
        else:
            label_ajuste = None

        plt.plot(f_fit, Vs_fit, "-", color=color,
                 label=(f"{nombre} ajuste\n"
                        f"m = {m:.2e}, b = {b:.2f}\n"
                        f"χ²r = {chi2_red:.2f}, R² = {r2:.3f}" if label_ajuste is None
                        else f"{label_ajuste}\n{nombre} ajuste\nm = {m:.2e}, b = {b:.2f}\nχ²r = {chi2_red:.2f}, R² = {r2:.3f}"))

    # Decoración
    titulo = f"Ajuste ODR: Voltaje vs Frecuencia {label}" if label else "Ajuste ODR: Voltaje vs Frecuencia"
    plt.title(titulo, fontsize=14)
    plt.xlabel("Frecuencia f (Hz)", fontsize=12)
    plt.ylabel("Voltaje Vs (V)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.show()

    return resultados

# =======================
# Datos en array numpy
# =======================
# Columnas: [Frecuencia, Sin filtro, Con filtro, Filtro PASCO]
data = np.array([
    [8.20264E+14, 1.76, 1.76, 1.76],
    [7.40858E+14, 1.52, 1.52, 1.52],
    [6.87858E+14, 1.31, 1.31, 1.31],
    [5.48996E+14, 0.92, 0.76, 0.69],
    [5.18672E+14, 0.95, 0.60, 0.59]
])
colores_orden1 = ["Ultra-Violeta", "Violeta", "Azul", "Verde", "Amarillo"]

# Extraer columnas
frecuencias = data[:, 0]
sin_filtro  = data[:, 1]
con_filtro  = data[:, 2]
filtro_pasco = data[:, 3]

# =======================
# Llamada a la función para los tres datasets
# =======================
resultados = analisis_tres_series(frecuencias, sin_filtro, con_filtro, filtro_pasco, label="Orden 1", nombres_colores=colores_orden1)




# =======================
# Datos en array numpy (Orden 2)
# =======================
# Columnas: [Frecuencia, Sin filtro, Con filtro, Filtro PASCO]
data_orden2 = np.array([
    [8.20264E+14, 1.60, 1.76, 1.76],
    [7.40858E+14, 1.25, 1.52, 1.52],
    [6.87858E+14, 0.99, 0.99, 0.99],
    [5.48996E+14, 0.99, 0.66, 0.37],
    [5.18672E+14, 0.86, 0.36, 0.33]
])

frecuencias2   = data_orden2[:, 0]
sin_filtro2    = data_orden2[:, 1]
con_filtro2    = data_orden2[:, 2]
filtro_pasco2  = data_orden2[:, 3]

# =======================
# Ejemplo de uso con analisis_rescalado
# =======================

# Ajuste Orden 2
resultados_orden2 = analisis_tres_series(frecuencias2, sin_filtro2, con_filtro2, filtro_pasco2, label="Orden 2")


# =======================
# Datos en array numpy (Orden -1)
# =======================
# Columnas: [Frecuencia, Sin filtro, Con filtro, Filtro PASCO]
data_ordenm1 = np.array([
    [8.20264E+14, 1.69, 1.69, 1.69],
    [7.40858E+14, 1.46, 1.46, 1.46],
    [6.87858E+14, 1.25, 1.25, 1.25],
    [5.48996E+14, 0.92, 0.75, 0.66],
    [5.18672E+14, 0.96, 0.50, 0.57]
])

# Extraer columnas
frecuencias_m1  = data_ordenm1[:, 0]
sin_filtro_m1   = data_ordenm1[:, 1]
con_filtro_m1   = data_ordenm1[:, 2]
filtro_pasco_m1 = data_ordenm1[:, 3]

# =======================
# Llamada a la función para los tres datasets
# =======================
resultados_ordenm1 = analisis_tres_series(frecuencias_m1, sin_filtro_m1, con_filtro_m1, filtro_pasco_m1, label="Orden -1")


# =======================
# Datos en array numpy (Orden -2)
# =======================
# Columnas: [Frecuencia, Sin filtro, Con filtro, Filtro PASCO]
data_ordenm2 = np.array([
    [8.20264E+14, 1.76, 1.76, 1.76],
    [7.40858E+14, 1.39, 1.39, 1.39],
    [6.87858E+14, 1.18, 1.18, 1.18],
    [5.48996E+14, 1.47, 1.26, 0.59],
    [5.18672E+14, 1.15, 0.51, 0.50]
])

# Extraer columnas
frecuencias_m2  = data_ordenm2[:, 0]
sin_filtro_m2   = data_ordenm2[:, 1]
con_filtro_m2   = data_ordenm2[:, 2]
filtro_pasco_m2 = data_ordenm2[:, 3]

# =======================
# Llamada a la función para los tres datasets
# =======================
resultados_ordenm2 = analisis_tres_series(frecuencias_m2, sin_filtro_m2, con_filtro_m2, filtro_pasco_m2, label="Orden -2")
