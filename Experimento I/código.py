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

# Modelos lineales
def f0(beta, x):
    m = beta[0]
    return m * x

def f1(beta, x):
    m, b = beta
    return m * x + b


def analisis_comparado(serie=1, con_residuos=True):
    # Selección de datos
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

    # Errores
    errI = 0.03 * I + 0.003
    errV = 0.005 * V + 100e-3
    errR = np.full_like(R, 0.003)

    B = cteB * I

    # Variables según serie
    if serie == 1:
        x = 1.0 / (B**2)
        y = R**2
        errx = (2 * cteB * errI) / (B**3)
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

    # --- Ajustes ---
    # Sin ordenada
    modelo0 = Model(f0)
    odr0 = ODR(RealData(x, y, sx=errx, sy=erry), modelo0, beta0=[1.0])
    out0 = odr0.run()
    m0, sm0 = out0.beta[0], out0.sd_beta[0]
    y_pred0 = f0(out0.beta, x)

    # Con ordenada
    modelo1 = Model(f1)
    odr1 = ODR(RealData(x, y, sx=errx, sy=erry), modelo1, beta0=[1.0, 0.0])
    out1 = odr1.run()
    m1, b1 = out1.beta
    sm1, sb1 = out1.sd_beta
    y_pred1 = f1(out1.beta, x)

    # Estadísticos
    def stats(y, y_pred, erry, p):
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        chi2 = np.sum(((y - y_pred) / erry)**2)
        dof = max(1, N - p)
        return r2, chi2/dof, chi2, dof

    r2_0, chi2red_0, chi2_0, dof0 = stats(y, y_pred0, erry, 1)
    r2_1, chi2red_1, chi2_1, dof1 = stats(y, y_pred1, erry, 2)

    # --- Gráfico con residuos ---
    if con_residuos:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})

        # Datos + ajustes
        ax1.errorbar(x, y, xerr=errx, yerr=erry, fmt='o',
                     ecolor="gray", capsize=3, label="Datos")
        xs = np.linspace(x.min(), x.max(), 200)
        ax1.plot(xs, f0([m0], xs), "r-", 
                 label=(f"Sin ordenada: m={m0:.3e}\n"
                        f"R²={r2_0:.4f}, χ²_red={chi2red_0:.3f}"))
        ax1.plot(xs, f1([m1, b1], xs), "b--", 
                 label=(f"Con ordenada: m={m1:.3e}, b={b1:.2e}\n"
                        f"R²={r2_1:.4f}, χ²_red={chi2red_1:.3f}"))
        ax1.set_ylabel(ylabel)
        ax1.legend(loc="best", fontsize=9)

        # Residuos en mismo gráfico
        resid0 = y - y_pred0
        resid1 = y - y_pred1
        ax2.errorbar(x, resid0, yerr=erry, fmt='o', color="red",
                     ecolor="lightcoral", capsize=3, label="Residuos sin ordenada")
        ax2.errorbar(x, resid1, yerr=erry, fmt='s', color="blue",
                     ecolor="lightblue", capsize=3, label="Residuos con ordenada")
        ax2.axhline(0, color='k', linestyle='--')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Residuos")
        ax2.legend(fontsize=9)

        plt.tight_layout()
        plt.show()

    # --- Resultados ---
    print(f"=== Serie {serie} ===")
    print("Modelo sin ordenada:")
    print(f"  m = {m0:.6e} ± {sm0:.2e}")
    print(f"  R² = {r2_0:.4f}, χ²_red = {chi2red_0:.3f}")

    print("\nModelo con ordenada:")
    print(f"  m = {m1:.6e} ± {sm1:.2e}, b = {b1:.3e} ± {sb1:.2e}")
    print(f"  R² = {r2_1:.4f}, χ²_red = {chi2red_1:.3f}")

    return {
        "sin_ordenada": {"m": m0, "sm": sm0, "r2": r2_0, "chi2_red": chi2red_0},
        "con_ordenada": {"m": m1, "sm": sm1, "b": b1, "sb": sb1, "r2": r2_1, "chi2_red": chi2red_1}
    }




# Ejecución de ejemplo
res1 = analisis_comparado(serie=1, con_residuos=True)
res2 = analisis_comparado(serie=2, con_residuos=True)
res3 = analisis_comparado(serie=3, con_residuos=True)
