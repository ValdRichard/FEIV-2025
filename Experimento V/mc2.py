import matplotlib.pyplot as plt
import os
import numpy as np

def graficar_masas_fotopicos_ponderado_con_error(rayos_gamma, 
                                                 mc2_rel, err_rel, 
                                                 mc2_no_rel, err_no_rel,
                                                 xlabel="Energía fotopico (keV)",
                                                 ylabel="$m_0c^2$ (keV)",
                                                 nombre_archivo="Masas_fotopicos_promedio_ponderado_error",
                                                 mostrar_grafica=True):
    """
    Grafica mc^2 relativista y no relativista con errores y muestra el promedio ponderado con su error.
    """
    if mostrar_grafica:
        plt.figure(figsize=(10,6))
        
        # Scatter relativista con barras de error
        plt.errorbar(rayos_gamma, mc2_rel, yerr=err_rel,
                     fmt='o', color='blue', alpha=1, capsize=3, label='$m_0c^2$ relativista')
        
        # Scatter no relativista con barras de error
        plt.errorbar(rayos_gamma, mc2_no_rel, yerr=err_no_rel,
                     fmt='o', color='red', alpha=1, capsize=3, label='$m_0c^2$ no relativista')
        
        # Promedios ponderados
        w_rel = 1 / err_rel**2
        promedio_rel = np.sum(mc2_rel * w_rel) / np.sum(w_rel)
        error_promedio_rel = np.sqrt(1 / np.sum(w_rel))
        
        w_no_rel = 1 / err_no_rel**2
        promedio_no_rel = np.sum(mc2_no_rel * w_no_rel) / np.sum(w_no_rel)
        error_promedio_no_rel = np.sqrt(1 / np.sum(w_no_rel))
        
        # Líneas horizontales de promedio ponderado
        plt.axhline(promedio_rel, color='blue', linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'Promedio ponderado relativista = {promedio_rel:.2f} ± {error_promedio_rel:.2f}')
        plt.axhline(promedio_no_rel, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                    label=f'Promedio ponderado no relativista = {promedio_no_rel:.2f} ± {error_promedio_no_rel:.2f}')
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Guardar imagen
        carpeta = "./Experimento V/Imagenes/Masas_fotopicos"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)
        
        plt.show()


# === Datos ===
rayos_gamma = np.array([1173.2, 1332.5, 666.5, 511, 1244, 299.6, 363])

mc2_rel = np.array([545.1929412, 595.8930211, 415.9064961, 500.5714286,
                    392.4765007, 392.6249724, 517.1037736])
err_rel = np.array([9.571465984, 47.99762383, 3.679719903, 11.89616949,
                    32.59663562, 32.80781881, 23.5797673])

mc2_no_rel = np.array([1021.192941, 1140.393021, 669.9064961, 672.0714286,
                       929.7265007, 483.1249724, 623.1037736])
err_no_rel = np.array([9.096746586, 47.49854334, 3.216770002, 11.52783072,
                       32.10329477, 32.72801979, 23.08247469])

# === Graficar ===
graficar_masas_fotopicos_ponderado_con_error(rayos_gamma, mc2_rel, err_rel, mc2_no_rel, err_no_rel)
