import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from matplotlib.lines import Line2D
# =======================
# Parámetros conocidos
# =======================
lambda0 = 9.50e-07          # Longitud de onda [m]
c = 299792458               # Velocidad de la luz [m/s]
f = 3.16e14                 # Frecuencia de la onda [Hz]
k = 1.3807*10**(-23)
# =======================
# Función lineal con ordenada al origen
# =======================
def f_lineal(beta, x):
    m, b = beta
    return m * x + b

modelo = Model(f_lineal)

# =======================
# Función de análisis
# =======================
def analisis(Tcrudo, Vcrudo, errTCrudo=None, errVCrudo=None):
    Tcrudo = np.asarray(Tcrudo, dtype=float)
    Vcrudo = np.asarray(Vcrudo, dtype=float)

    # errores
    if errTCrudo is None:
        errT = np.full_like(Tcrudo, 1e-9)
    else:
        errT = np.asarray(errTCrudo, dtype=float) / (Tcrudo**2)

    if errVCrudo is None:
        errV = np.full_like(Vcrudo, 0.01)
    else:
        errV = np.asarray(errVCrudo, dtype=float) / Vcrudo

    # variables transformadas
    T = 1.0 / Tcrudo
    V = np.log(Vcrudo)

    # ODR
    data = RealData(T, V, sx=errT, sy=errV)
    beta0 = [1.0, 0.0]
    odr = ODR(data, modelo, beta0=beta0)
    out = odr.run()

    m, b = out.beta
    sm, sb = out.sd_beta
    y_pred = f_lineal(out.beta, T)

    # R²
    ss_res = np.sum((V - y_pred) ** 2)
    ss_tot = np.sum((V - np.mean(V)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # h
    planck = -(m * k) / f
    splanck = (k * sm) / f

    # --- ploteo ---
    plt.figure(figsize=(9, 6))

    # puntos con errorbar → este handle va a la leyenda
    puntos = plt.errorbar(T, V, xerr=errT, yerr=errV,
                          fmt='o', color="orange", ecolor="gray",
                          capsize=3, label="Puntos experimentales")

    # línea de ajuste (proxy para leyenda)
    plt.plot(T, y_pred, color='red', linewidth=1.5)
    line_proxy = Line2D([0], [0], color='red', linewidth=1.5)

    # texto extra (proxy vacío)
    text_proxy = Line2D([], [], linestyle='None')

    # labels
    label_line   = f"Ajuste lineal: y = {m:.2e} x + {b:.2f}\nR² = {r2:.4f}"
    label_htext  = f"h = ({planck:.3e} ± {splanck:.1e}) J·s"

    # leyenda
    plt.legend(handles=[puntos, line_proxy, text_proxy],
               labels=[puntos.get_label(), label_line, label_htext],
               loc='best', fontsize=12)

    plt.xlabel("1/T (1/K)")
    plt.ylabel("ln(V)")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.grid(True, which="both", linestyle="--", alpha=0.4)  # grillado

    plt.tight_layout()
    plt.show()

    # salida consola
    print("Resultados del ajuste:")
    print(f"Pendiente m = {m:.5e} ± {sm:.4e}")
    print(f"Ordenada b  = {b:.5e} ± {sb:.1e}")
    print(f"Planck h = {planck:.5e} ± {splanck:.1e}")

# Lampara 1 45W
# Voltaje (V)

# =========================
# Lámpara 1 - 45W
# =========================
V = np.array([0.176, 0.268, 0.383, 0.514, 0.681,
              0.876, 1.081, 1.563, 1.843, 2.468, 2.805])
errV = np.array([0.00728, 0.01004, 0.01349, 0.01742, 0.02243,
                 0.02828, 0.03443, 0.04889, 0.05729, 0.07604, 0.08615])
T = np.array([1739.661181, 1831.262967, 1917.775017, 2002.038683, 2072.332904,
              2140.018044, 2208.344901, 2334.020642, 2391.106121, 2500.65497, 2541.56139])
errT = np.array([43.3, 45.3, 47.2, 49.1, 50.7,
                 52.2, 53.8, 56.6, 58.0, 60.5, 61.4])
analisis(T, V, errT, errV)

# =========================
# Lámpara 2 - 60W
# =========================
V = np.array([0.288, 0.435, 0.616, 0.838, 1.077, 1.359,
              1.675, 2.037, 2.386, 2.806, 3.278, 3.716, 4.217])
errV = np.array([0.01064, 0.01505, 0.02048, 0.02714, 0.03431, 0.04277,
                 0.05225, 0.06311, 0.07358, 0.08618, 0.10034, 0.11348, 0.12851])
T = np.array([1750.89558, 1784.244081, 1921.115603, 1993.349518,
              2068.611716, 2135.660416, 2200.460202, 2241.900012, 2321.388777,
              2375.146919, 2426.108409, 2480.28932, 2528.959828])
errT = np.array([43.6, 44.1, 47.3, 48.9, 50.6, 52.1,
                 53.6, 54.5, 56.3, 57.6, 58.7, 60.0, 61.1])
analisis(T, V, errT, errV)

# =========================
# Lámpara 3 - 75W
# =========================
V = np.array([0.395, 0.594, 0.824, 1.105, 1.42, 1.774,
              2.172, 2.605, 3.099, 3.625, 4.169, 4.734, 5.38])
errV = np.array([0.01385, 0.01982, 0.02672, 0.03515, 0.0446, 0.05522,
                 0.06716, 0.08015, 0.09497, 0.11075, 0.12707, 0.14402, 0.1634])
T = np.array([1846.7, 1933.2, 2019.8, 2096.6, 2170.2, 2240.8,
              2308.4, 2371.6, 2432.2, 2487.8, 2546.8, 2602.3, 2653.2])
errT = np.array([46.0, 47.8, 49.7, 51.4, 53.1, 54.7,
                 56.2, 57.7, 59.0, 60.3, 61.7, 63.0, 64.1])
analisis(T, V, errT, errV)

