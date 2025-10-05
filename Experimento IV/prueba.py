import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

def graficar(df):
    plt.figure(figsize=(8,5))
    plt.scatter(df["Canal"], df["Cuentas"], marker='.')
    plt.xlabel("Canal")
    plt.ylabel("Cuentas")
    plt.title("Espectro calibrado")
    plt.grid(alpha=0.3)
    plt.show()

def leer_spe(path, nombre):
    with open(path + nombre, "r") as f:
        lines = [l.strip() for l in f]

    start = lines.index("$DATA:") + 10
    end = lines.index("$ROI:")
    counts = [int(x) for x in lines[start:end] if x]

    df = pd.DataFrame({
        "Canal": range(len(counts)),
        "Cuentas": counts
    })
    return df


def espectro(path, nombre):
    df = leer_spe(path, nombre)
    fondo = leer_spe(path, 'Fondo-17-9-4700.Spe')
    df["Cuentas"] = df["Cuentas"] - fondo["Cuentas"]
    return df



# ==============================
#  Definición de la Gaussiana
# ==============================
def funcion_gaussiana(beta, x):
    """
    Función gaussiana para ODR.
    beta[0] = amplitud
    beta[1] = media
    beta[2] = sigma
    beta[3] = pendiente
    beta[4] = ordenada
    """
    return beta[0] * np.exp(-(x - beta[1])**2 / (2 * beta[2]**2)) + beta[3] * x + beta[4]


# ==============================
#  Ajuste con ODR
# ==============================
def ajustar_gaussiana_odr(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True):
    # Valores iniciales por defecto
    if p0 is None:
        p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
    
    # Manejo de errores si no se pasan
    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)  # 1% del rango
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)  # 1% del rango
    
    # Crear el modelo para ODR
    modelo_gauss = Model(funcion_gaussiana)
    
    # Crear datos con errores
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    
    # Configurar ODR
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    
    # Ejecutar el ajuste
    output = odr.run()
    
    # Extraer resultados
    parametros = output.beta
    errores = output.sd_beta
    
    # Función ajustada
    def gaussiana_ajustada(x):
        return funcion_gaussiana(parametros, x)
    
    # Mostrar gráfica
    if mostrar_grafica:
        plt.figure(figsize=(10,6))
        plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, 
                     fmt='o', alpha=0.7, label='Datos', 
                     color='blue', capsize=3)
        
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 1000)
        y_fit = gaussiana_ajustada(x_fit)
        
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                 label=(f'Gaussiana ODR\n'
                        f'A={parametros[0]:.2f}±{errores[0]:.2f}\n'
                        f'μ={parametros[1]:.2f}±{errores[1]:.2f}\n'
                        f'σ={parametros[2]:.2f}±{errores[2]:.2f}'))
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Ajuste Gaussiano con ODR')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
    return parametros, errores, output, gaussiana_ajustada


# --- Ejemplo de uso ---
ruta = "./Experimento IV/Datos/"
df = espectro(ruta, 'Cs137-Pb.Spe')

graficar(df[560:800])
x_data = df["Canal"][560:800]
x_err = np.ones_like(1/1024)
y_data = df["Cuentas"][560:800]
y_err = np.sqrt(y_data)

# Ajuste
parametros, errores, output, gauss_ajustada = ajustar_gaussiana_odr(
    x_data, y_data, x_err, y_err, 
    # beta[0] = amplitud
    # beta[1] = media
    # beta[2] = sigma
    # beta[3] = pendiente
    # beta[4] = ordenada
    p0=[0,662,7,4,0]
)

print("Parámetros ajustados:", parametros)
print("Errores estándar:", errores)

# x_data = df["Canal"][10:60]
# x_err = np.ones_like(1/1024)
# y_data = df["Cuentas"][10:60]
# y_err = np.sqrt(y_data)

# # Ajuste
# parametros, errores, output, gauss_ajustada = ajustar_gaussiana_odr(
#     x_data, y_data, x_err, y_err, 
#     # beta[0] = amplitud
#     # beta[1] = media
#     # beta[2] = sigma
#     # beta[3] = pendiente
#     # beta[4] = ordenada
#     p0=[0,30,7,4,0]
# )

# print("Parámetros ajustados:", parametros)
# print("Errores estándar:", errores)