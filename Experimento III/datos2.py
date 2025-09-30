import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# ----------------------------
# Ajuste lineal ODR
# ----------------------------
def fit_lineal(Vs, f, scale=1e14):
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

    # Parámetros ajustados
    m_scaled, b = out.beta
    sm_scaled, sb = out.sd_beta
    m, sm = m_scaled / scale, sm_scaled / scale

    # --- Métricas ---
    y_pred = f_lineal(out.beta, f_scaled)
    chi2 = np.sum(((Vs - y_pred) / errV) ** 2)
    dof = len(Vs) - len(out.beta)
    chi2_red = chi2 / dof if dof > 0 else np.nan
    ss_res = np.sum((Vs - y_pred) ** 2)
    ss_tot = np.sum((Vs - np.mean(Vs)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return m, sm, b, sb, chi2_red, r2


# ----------------------------
# Función para graficar las 3 series
# ----------------------------
def analisis_tres_series(f, sin_filtro, con_filtro, pasco, scale=1e14, label="", nombres_colores=None):
    datasets = {
        "Datos - sin filtro": sin_filtro,
        'Datos - con filtro "a"': con_filtro,
        'Datos - con filtro "b"': pasco
    }

    colores = ["tab:blue", "tab:orange", "tab:green"]
    marcadores = ["D", "s", "v"]       
    tamanios = [6, 6, 6]               

    plt.figure(figsize=(9, 6))

    resultados = {}
    funcion_label_agregada = False

    for (nombre, Vs), color, marker, msize in zip(datasets.items(), colores, marcadores, tamanios):
        f_plot, Vs_plot = f.copy(), Vs.copy()   # todos los puntos (para graficar)

        # --- EXCLUSIÓN SOLO EN ORDEN -2 y solo en "con filtro a" ---
        if label == "Orden -2" and nombre == 'Datos - con filtro "a"':
            mask = ~((np.isclose(Vs, 1.26)) & (np.isclose(f, 5.48996E+14)))
            f_fit, Vs_fit = f[mask], Vs[mask]
        else:
            f_fit, Vs_fit = f, Vs

        # Ajuste
        m, sm, b, sb, chi2_red, r2 = fit_lineal(Vs_fit, f_fit, scale=scale)
        resultados[nombre] = {"m": m, "sm": sm, "b": b, "sb": sb, "chi2_red": chi2_red, "R2": r2}

        # Etiquetas
        if not funcion_label_agregada:
            label_ajuste = f"y = m·x + b"
            funcion_label_agregada = True
        else:
            label_ajuste = None

        # Dibujar puntos con error
        errV = 0.03 * np.array(Vs_plot) + 0.01
        errF = np.full_like(f_plot, 1e8)
        plt.errorbar(f_plot, Vs_plot, yerr=errV, xerr=errF,
                     fmt=marker, markersize=msize,
                     ecolor="gray", capsize=3,
                     label=(f"{nombre} ajuste\n"
                            f"m = ({m:.2e} ± {sm:.1e})\n"
                            f"b = ({b:.2f} ± {sb:.2f})\n"
                            f"χ²r = {chi2_red:.2f}, R² = {r2:.4f}" if label_ajuste is None
                            else f"{label_ajuste}\n{nombre} ajuste\nm = {m:.2e}, b = {b:.2f}\nχ²r = {chi2_red:.2f}, R² = {r2:.3f}")
                     , alpha=0.8, color=color)

        # Dibujar ajuste
        f_line = np.linspace(min(f), max(f), 200)
        plt.plot(f_line, m*f_line + b, color=color)

        # Si es el punto excluido, marcarlo con recuadro punteado en naranja
        if label == "Orden -2" and nombre == 'Datos - con filtro "a"':
            excluded_x, excluded_y = 5.48996E+14, 1.26
            plt.scatter([excluded_x], [excluded_y], color=color, zorder=5)
            plt.gca().add_patch(
                plt.Rectangle((excluded_x-0.1e14, excluded_y-0.1),
                              0.2e14, 0.2,
                              fill=False, edgecolor=color, linestyle="--", linewidth=2,
                              label="Punto excluido")
            )

    # Decoración
    plt.title('', fontsize=14)
    plt.xlabel("Frecuencia f (Hz)", fontsize=12)
    plt.ylabel("Voltaje Vs (V)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc="best")
    plt.tight_layout()
    plt.show()

    return resultados


# ----------------------------
# Ejemplo de uso en Orden -2
# ----------------------------
data_ordenm2 = np.array([
    [8.20264E+14, 1.76, 1.76, 1.76],
    [7.40858E+14, 1.39, 1.39, 1.39],
    [6.87858E+14, 1.18, 1.18, 1.18],
    [5.48996E+14, 1.47, 1.26, 0.59],  
    [5.18672E+14, 1.15, 0.51, 0.50]
])
frecuencias_m2  = data_ordenm2[:, 0]
sin_filtro_m2   = data_ordenm2[:, 1]
con_filtro_m2   = data_ordenm2[:, 2]
filtro_pasco_m2 = data_ordenm2[:, 3]

resultados_ordenm2 = analisis_tres_series(frecuencias_m2, sin_filtro_m2, con_filtro_m2, filtro_pasco_m2, label="Orden -2")
