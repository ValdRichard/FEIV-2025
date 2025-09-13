import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# Parámetros experimentales
n = 130                      # Número de vueltas
mu0 = 4 * np.pi * 1e-7       # Permeabilidad magnética
a = 0.15                     # Radio de las bobinas (m)
cteB = (4/5)**(3/2) * mu0 * n / a

# Orden de las columnas
# I - V -R
# Serie 1

# V - cte
serie1 = np.array([
    [1.244000, 250.000000, 0.053000],
    [1.302000, 250.000000, 0.052000],
    [1.393000, 250.000000, 0.049250],
    [1.540000, 250.000000, 0.046000],
    [1.676000, 250.000000, 0.042000],
])


# Serie 2

# I - cte
serie2 = np.array([
    [1.506000, 270.000000, 0.048250],
    [1.506000, 250.000000, 0.046000],
    [1.506000, 230.000000, 0.043500],
    [1.506000, 210.000000, 0.041500],
    [1.506000, 190.000000, 0.040000],
    [1.506000, 170.000000, 0.038000],
])

# Serie 3

# R - cte
serie3 = np.array([
    [1.500000, 200.000000, 0.039500],
    [1.578000, 220.000000, 0.039500],
    [1.488000, 180.000000, 0.039500],
])


# Modelo lineal y función auxiliar
def f(beta, x):
    m = beta[0]
    return m * x
def analisis(serie=1, con_residuos=True):
    modelo = Model(f)

    # beta0 inicial por serie
    if serie == 3:
        beta0 = [1e8]
    else:
        beta0 = [1.0]

    # Selección de datos según la serie
    if serie == 1:
        datos = serie1
    elif serie == 2:
        datos = serie2
    elif serie == 3:
        datos = serie3
    else:
        raise ValueError("serie debe ser 1, 2 o 3.")

    I = datos[:, 0]
    V = datos[:, 1]
    R = datos[:, 2]

    N = len(V)
    if N == 0:
        raise ValueError("No hay datos para la serie solicitada.")

    # === ERRORES ===
    errI = 0.03 * I + 0.003
    errV = 0.005 * V + 100e-3
    errR = np.full_like(R, 0.003)   # error fijo de 3 mm (0.003 m)

    # Campo magnético (B) a partir de la corriente
    B = cteB * I

    # --- selecciono x,y según serie ---
    if serie == 1:
        x = 1.0 / (B**2)
        y = R**2
        errx = (2 * errI / I) * x
        erry = 2 * R * errR
        xlabel, ylabel = "1/B² (T⁻²)", "R² (m²)"
    elif serie == 2:
        x = V
        y = R**2
        errx = errV
        erry = 2 * R * errR
        xlabel, ylabel = "V (V)", "R² (m²)"
    elif serie == 3:
        x = B**2
        y = V
        errx = 2 * B * cteB * errI
        erry = errV
        xlabel, ylabel = "B² (T²)", "V (V)"

    # --- Ajuste ODR ---
    data = RealData(x, y, sx=errx, sy=erry)
    odr = ODR(data, modelo, beta0=beta0)
    out = odr.run()

    m = float(out.beta[0])
    sm = float(out.sd_beta[0])

    # predicción y residuos
    y_pred = f([m], x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan

    chi2 = np.sum(((y - y_pred) / erry)**2)
    dof = max(1, N - len(out.beta))
    chi2_red = chi2 / dof

    # --- Cálculo de e/m ---
    sVmean = errV.mean() / np.sqrt(N)
    sImean = errI.mean() / np.sqrt(N)
    sRmean = errR.mean() / np.sqrt(N)

    if serie == 1:
        Vmean = V.mean()
        em = 2.0 * Vmean / m
        d_em_dm = -2.0 * Vmean / (m**2)
        d_em_dV = 2.0 / m
        errem = np.sqrt((d_em_dm * sm)**2 + (d_em_dV * sVmean)**2)
    elif serie == 2:
        Imean = I.mean()
        Bmean = cteB * Imean
        em = 2.0 / (m * (Bmean**2))
        d_em_dm = -2.0 / (m**2 * Bmean**2)
        d_em_dI = -4.0 / (m * (cteB**2) * (Imean**3))
        errem = np.sqrt((d_em_dm * sm)**2 + (d_em_dI * sImean)**2)
    elif serie == 3:
        Rmean = R.mean()
        em = 2.0 * m / (Rmean**2)
        d_em_dm = 2.0 / (Rmean**2)
        d_em_dR = -4.0 * m / (Rmean**3)
        errem = np.sqrt((d_em_dm * sm)**2 + (d_em_dR * sRmean)**2)

    # --- Gráficos ---
    if con_residuos:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Ajuste principal
        ax1.errorbar(x, y, xerr=errx, yerr=erry, fmt='o', ecolor="gray",
                     capsize=3, label="Datos")
        sort_idx = np.argsort(x)
        xs = x[sort_idx]
        ax1.plot(xs, f([m], xs), "r-", label="Ajuste ODR")
        ax1.set_ylabel(ylabel)

        textstr = (
            f"Función: y = m·x\n"
            f"m = ({m:.3e} ± {sm:.1e})\n"
            f"R² = {r2:.4f}\n"
            f"χ²_red = {chi2_red:.3f}")
        ax1.text(0.3, 0.98, textstr, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax1.legend()

        # Gráfico de residuos
        residuos = y - y_pred
        ax2.errorbar(x, residuos, yerr=erry, fmt='o', ecolor="gray", capsize=3)
        ax2.axhline(0, color='r', linestyle='--')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Residuos")
    else:
        plt.errorbar(x, y, xerr=errx, yerr=erry, fmt='o', ecolor="gray",
                     capsize=3, label="Datos")
        sort_idx = np.argsort(x)
        xs = x[sort_idx]
        plt.plot(xs, f([m], xs), "r-", label="Ajuste ODR")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        textstr = f"R² = {r2:.4f}\nχ²_red = {chi2_red:.3f}"
        plt.gca().text(0.3, 0.98, textstr, transform=plt.gca().transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"Serie {serie}")
    print(f"Pendiente m = {m:.6e} ± {sm:.2e}")
    print(f"e/m = {em:.6e} ± {errem:.2e} C/kg")
    print(f"R² = {r2:.6f}")
    print(f"χ² reducido = {chi2_red:.4f} (chi2 = {chi2:.3f}, dof = {dof})")

    return {"m": m, "sm": sm, "em": em, "serrem": errem,
            "r2": r2, "chi2_red": chi2_red}


# Ejecución de ejemplo:
res1 = analisis(serie=1, con_residuos=True)
res2 = analisis(serie=2, con_residuos=True)  
res2 = analisis(serie=3, con_residuos=True)  

