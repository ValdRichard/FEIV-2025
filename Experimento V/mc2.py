import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.odr import ODR, Model, RealData

def graficar_masas_fotopicos_ajuste_con_ordenada(
    rayos_gamma, err_gamma,
    mc2_rel, err_rel, 
    mc2_no_rel, err_no_rel,
    xlabel="Energía del fotopico (keV)",
    ylabel="$m_0 c^2$ (keV)",
    nombre_archivo="Masas_fotopicos_ajuste_con_ordenada",
    mostrar_grafica=True):

    # Modelo lineal con ordenada: y = a*x + b
    def modelo_lineal(B, x):
        return B[0] * x + B[1]

    model = Model(modelo_lineal)

    # === AJUSTE RELATIVISTA ===
    data_rel = RealData(rayos_gamma, mc2_rel, sx=err_gamma, sy=err_rel)
    odr_rel = ODR(data_rel, model, beta0=[0.5, 100])
    out_rel = odr_rel.run()
    a_rel, b_rel = out_rel.beta
    aerr_rel, berr_rel = out_rel.sd_beta

    # === AJUSTE NO RELATIVISTA ===
    data_no = RealData(rayos_gamma, mc2_no_rel, sx=err_gamma, sy=err_no_rel)
    odr_no = ODR(data_no, model, beta0=[1.0, 100])
    out_no = odr_no.run()
    a_no, b_no = out_no.beta
    aerr_no, berr_no = out_no.sd_beta

    # === GRAFICAR ===
    if mostrar_grafica:
        plt.figure(figsize=(10,6))

        # Datos relativistas
        plt.errorbar(rayos_gamma, mc2_rel, xerr=err_gamma, yerr=err_rel,
                     fmt='o', color='blue', capsize=3, label='Datos relativistas')

        # Datos no relativistas
        plt.errorbar(rayos_gamma, mc2_no_rel, xerr=err_gamma, yerr=err_no_rel,
                     fmt='o', color='red', capsize=3, label='Datos no relativistas')

        # Rectas ajustadas
        xline = np.linspace(np.min(rayos_gamma)*0.9, np.max(rayos_gamma)*1.1, 400)

        # Relativista
        plt.plot(
            xline, a_rel * xline + b_rel, 'b--',
            label=(
                rf"Relativista: $y = aE + b$\n"
                rf"$a = {a_rel:.4f}\pm{aerr_rel:.4f}$, "
                rf"$b = {b_rel:.1f}\pm{berr_rel:.1f}$"
            )
        )

        # No relativista
        plt.plot(
            xline, a_no * xline + b_no, 'r--',
            label=(
                rf"No relativista: $y = aE + b$\n"
                rf"$a = {a_no:.4f}\pm{aerr_no:.4f}$, "
                rf"$b = {b_no:.1f}\pm{berr_no:.1f}$"
            )
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Guardar
        carpeta = "./Experimento V/Imagenes/Masas_fotopicos"
        os.makedirs(carpeta, exist_ok=True)
        plt.savefig(f"{carpeta}/{nombre_archivo}.png", dpi=300)

        plt.show()

    return (a_rel, aerr_rel, b_rel, berr_rel), (a_no, aerr_no, b_no, berr_no)



# === DATOS ===
rayos_gamma = np.array([1173.2, 1332.5, 666.5, 511, 1244, 299.6, 363])

err_gamma = np.array([1, 1, 0.4, 2, 2, 7, 0.5])

mc2_rel = np.array([545.1929412, 595.8930211, 415.9064961, 500.5714286,
                    392.4765007, 392.6249724, 517.1037736])

err_rel = np.array([9.571465984, 47.99762383, 3.679719903, 11.89616949,
                    32.59663562, 32.80781881, 23.5797673])

mc2_no_rel = np.array([1021.192941, 1140.393021, 669.9064961, 672.0714286,
                       929.7265007, 483.1249724, 623.1037736])

err_no_rel = np.array([9.096746586, 47.49854334, 3.216770002, 11.52783072,
                       32.10329477, 32.72801979, 23.08247469])

# === Ejecutar ===
graficar_masas_fotopicos_ajuste_con_ordenada(
    rayos_gamma, err_gamma,
    mc2_rel, err_rel, 
    mc2_no_rel, err_no_rel
)
