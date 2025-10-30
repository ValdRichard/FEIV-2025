import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.special import erf
def fit_lineal(x, y, err_x=None, err_y=None):
    # Conversión a arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Manejo de errores
    if err_x is None:
        err_x = np.zeros_like(x)
    elif np.isscalar(err_x):
        err_x = np.full_like(x, err_x, dtype=float)

    if err_y is None:
        err_y = np.zeros_like(y)
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
        


    plt.xlabel('Canal')
    plt.ylabel('Energía [keV]')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # # Chi² reducido
    # chi2 = np.sum(((y - y_pred) / np.maximum(err_y, 1e-12)) ** 2)
    # dof = len(y) - len(out.beta)
    # chi2_red = chi2 / dof if dof > 0 else np.nan

    return m, sm, b, sb
    # return m, sm, b, sb, chi2_red, r2



def graficar(x, y, xlabel, ylabel):
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.show()

def graficar_con_error(x, y, xerr, yerr, xlabel, ylabel, titulo=None):
    plt.figure(figsize=(8, 5))
    
    # Graficamos con barras de error
    plt.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt='o',                # formato del punto (o = círculo)
        ecolor='gray',          # color de las barras de error
        elinewidth=1,           # grosor de línea de error
        capsize=3,              # tamaño de los "topes" en las barras
        markersize=4,           # tamaño de los puntos
        markeredgecolor='black',
        markerfacecolor='blue',
        alpha=0.8
    )
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if titulo:
        plt.title(titulo)
    plt.grid(alpha=0.3)
    plt.tight_layout()
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
    fondo = leer_spe(path, 'Fondo-18-9-1800s.Spe')
    df["Cuentas"] = df["Cuentas"] - fondo["Cuentas"]
    return df


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

def funcion_gaussiana_doble(beta, x):
    """
    Función gaussiana para ODR.
    beta[0] = amplitud 1
    beta[1] = media 1
    beta[2] = sigma 1
    beta[3] = pendiente 
    beta[4] = ordenada 
    beta[5] = amplitud 2
    beta[6] = media 2
    beta[7] = sigma 2
    """
    return beta[0] * np.exp(-(x - beta[1])**2 / (2 * beta[2]**2)) + beta[3] * x + beta[4] + beta[5] * np.exp(-(x - beta[6])**2 / (2 * beta[7]**2))

def funcion_gaussiana_CUADRUPLE(beta, x):
    """
    Función gaussiana para ODR.
    beta[0] = amplitud 1
    beta[1] = media 1
    beta[2] = sigma 1
    beta[3] = pendiente 
    beta[4] = ordenada 
    beta[5] = amplitud 2
    beta[6] = media 2
    beta[7] = sigma 2
    """
    return beta[0] * np.exp(-(x - beta[1])**2 / (2 * beta[2]**2)) + beta[3] * x + beta[4] + beta[5] * np.exp(-(x - beta[6])**2 / (2 * beta[7]**2)) + beta[8] * np.exp(-(x - beta[9])**2 / (2 * beta[10]**2)) + beta[11] * np.exp(-(x - beta[12])**2 / (2 * beta[13]**2))


def funcion_borde_compton(beta, x):
    """
    Modelo del borde Compton (función tipo error con desplazamiento vertical).
    
    beta[0] = A      (amplitud)
    beta[1] = xc     (posición del borde)
    beta[2] = sigma  (ancho)
    beta[3] = y0     (desplazamiento vertical)
    """
    A, xc, sigma, y0 = beta
    z = (x - xc) / (np.sqrt(2) * sigma)
    return A * (1 - erf(z)) + y0


def ajustar_borde_compton(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True,
                          nombre_archivo="BordeCompton"):
    """
    Ajusta un borde Compton con ODR usando la función tipo error + y0.
    Guarda la imagen si mostrar_grafica=True en Imagenes/BordeCompton.
    """
    if p0 is None:
        A0 = np.max(y_data) - np.min(y_data)
        xc0 = x_data[np.argmax(np.gradient(y_data))]
        sigma0 = (np.max(x_data) - np.min(x_data)) / 20
        y0 = np.min(y_data)
        p0 = [A0, xc0, sigma0, y0]

    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)

    modelo_compton = Model(funcion_borde_compton)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_compton, beta0=p0)
    output = odr.run()

    parametros = output.beta
    errores = output.sd_beta

    def borde_compton_ajustada(x):
        return funcion_borde_compton(parametros, x)

    # Calcular R²
    y_pred = borde_compton_ajustada(x_data)
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - ss_res/ss_tot

    if mostrar_grafica:
        plt.figure(figsize=(10,6))
        plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, 
                     fmt='o', alpha=0.5, label='Datos', 
                     color='orange', capsize=3)
        
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 1000)
        y_fit = borde_compton_ajustada(x_fit)
        
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                 label=(f'Borde Compton ODR\n'
                        f'A={parametros[0]:.2f}±{errores[0]:.2f}\n'
                        f'E={parametros[1]:.2f}±{errores[1]:.2f}\n'
                        f'σ={parametros[2]:.2f}±{errores[2]:.2f}\n'
                        f'y0={parametros[3]:.2f}±{errores[3]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Energía [keV]')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento V/Imagenes/BordeCompton"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)

        plt.show()

    return parametros, errores, output, borde_compton_ajustada

def ajustar_gaussiana_odr(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True,
                          nombre_archivo="Gaussiana"):
    if p0 is None:
        p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
    
    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)
    
    modelo_gauss = Model(funcion_gaussiana)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    output = odr.run()
    
    parametros = output.beta
    errores = output.sd_beta
    
    def gaussiana_ajustada(x):
        return funcion_gaussiana(parametros, x)
    
    # Calcular R²
    y_pred = gaussiana_ajustada(x_data)
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - ss_res/ss_tot
    
    if mostrar_grafica:
        plt.figure(figsize=(10,6))
        plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, 
                     fmt='o', alpha=0.5, label='Datos', 
                     color='orange', capsize=3)
        
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 1000)
        y_fit = gaussiana_ajustada(x_fit)
        
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                 label=(f'Ajuste gaussiana\n'
                        f'A={parametros[0]:.2f}±{errores[0]:.2f}\n'
                        f'E={parametros[1]:.2f}±{errores[1]:.2f}\n'
                        f'σ={parametros[2]:.2f}±{errores[2]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Energía [keV]')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento V/Imagenes/Gaussiana"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)

        plt.show()
    
    return parametros, errores, output, gaussiana_ajustada

def ajustar_gaussiana_doble_odr(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True,
                          nombre_archivo="Gaussiana doble"):
    if p0 is None:
        p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
    
    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)
    
    modelo_gauss = Model(funcion_gaussiana_doble)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    output = odr.run()
    
    parametros = output.beta
    errores = output.sd_beta
    
    def gaussiana_ajustada(x):
        return funcion_gaussiana_doble(parametros, x)
    
    # Calcular R²
    y_pred = gaussiana_ajustada(x_data)
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - ss_res/ss_tot
    
    if mostrar_grafica:
        plt.figure(figsize=(10,6))
        plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, 
                     fmt='o', alpha=0.5, label='Datos', 
                     color='orange', capsize=3)
        
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 1000)
        y_fit = gaussiana_ajustada(x_fit)
        
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                 label=(f'Ajuste gaussiana\n'
                        f'A_1={parametros[0]:.2f}±{errores[0]:.2f}\n'
                        f'C_1={parametros[1]:.2f}±{errores[1]:.2f}\n'
                        f'σ_1={parametros[2]:.2f}±{errores[2]:.2f}\n'
                        f'A_2={parametros[5]:.2f}±{errores[5]:.2f}\n'
                        f'C_2={parametros[6]:.2f}±{errores[6]:.2f}\n'
                        f'σ_2={parametros[7]:.2f}±{errores[7]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Canal')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento V/Imagenes/Gaussiana doble"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)

        plt.show()
    
    return parametros, errores, output, gaussiana_ajustada

def ajustar_gaussiana_cuadruple_odr(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True,
                          nombre_archivo="Gaussiana cuadruple"):
    if p0 is None:
        p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
    
    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)
    
    modelo_gauss = Model(funcion_gaussiana_CUADRUPLE)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    output = odr.run()
    
    parametros = output.beta
    errores = output.sd_beta
    
    def gaussiana_ajustada(x):
        return funcion_gaussiana_CUADRUPLE(parametros, x)
    
    # Calcular R²
    y_pred = gaussiana_ajustada(x_data)
    ss_res = np.sum((y_data - y_pred)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r2 = 1 - ss_res/ss_tot
    
    if mostrar_grafica:
        plt.figure(figsize=(10,6))
        plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, 
                     fmt='o', alpha=0.5, label='Datos', 
                     color='orange', capsize=3)
        
        x_fit = np.linspace(np.min(x_data), np.max(x_data), 1000)
        y_fit = gaussiana_ajustada(x_fit)
        
        plt.plot(x_fit, y_fit, 'r-', linewidth=2, 
                 label=(f'Ajuste gaussiana\n'
                        f'C_1={parametros[1]:.2f}±{errores[1]:.2f}\n'
                        f'σ_1={parametros[2]:.2f}±{errores[2]:.2f}\n'
                        f'C_2={parametros[6]:.2f}±{errores[6]:.2f}\n'
                        f'σ_2={parametros[7]:.2f}±{errores[7]:.2f}\n'
                        f'C_3={parametros[9]:.2f}±{errores[9]:.2f}\n'
                        f'σ_3={parametros[10]:.2f}±{errores[10]:.2f}\n'
                        f'C_4={parametros[12]:.2f}±{errores[12]:.2f}\n'
                        f'σ_4={parametros[13]:.2f}±{errores[13]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Canal')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento V/Imagenes/Gaussiana doble"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)

        plt.show()
    
    return parametros, errores, output, gaussiana_ajustada


def calibrar(canal, sigma_canal, m, b, sm, sb):
    
    canal = np.array(canal, dtype=float)
    E = m * canal + b

    if sigma_canal is None:
        sigma_canal = np.zeros_like(canal, dtype=float)
    else:
        if np.isscalar(sigma_canal):
            sigma_canal = np.full_like(canal, float(sigma_canal))

    # Propagación: derivadas: dE/dm = canal, dE/db = 1, dE/dcanal = m
    # Var(E) ≈ (canal^2 * Var(m)) + Var(b) + (m^2 * Var(canal))
    sigma_E = np.sqrt((canal**2) * (sm**2) + (sb**2) + (m**2) * (sigma_canal**2))
    return E, sigma_E

def cortar_datos(izquierda, derecha, x, y, x_err, y_err):
    x_data = x[izquierda:derecha]
    y_data = y[izquierda:derecha]
    x_err = x_err[izquierda:derecha]
    y_err = y_err[izquierda:derecha]
    return x_data, y_data, x_err, y_err


def ajustar_pico_gaussiano(x_data, y_data, x_err, y_err, p0, mostrarGrafica, nombre_archivo = 'None'):
    """Ajusta un pico gaussiano con ODR"""
    parametros, errores, output, gauss_ajustada = ajustar_gaussiana_odr(
        x_data, y_data, x_err, y_err, p0=p0, mostrar_grafica=mostrarGrafica, nombre_archivo = nombre_archivo
    )
    return parametros, errores, output, gauss_ajustada

def ajustar_pico_gaussiano_doble(x_data, y_data, x_err, y_err, p0, mostrarGrafica, nombre_archivo = 'None'):
    """Ajusta un pico gaussiano doble con ODR"""
    parametros, errores, output, gauss_ajustada = ajustar_gaussiana_doble_odr(
        x_data, y_data, x_err, y_err, p0=p0, mostrar_grafica=mostrarGrafica, nombre_archivo = nombre_archivo
    )
    return parametros, errores, output, gauss_ajustada


def ajustar_pico_gaussiano_CUADRUPLE(x_data, y_data, x_err, y_err, p0, mostrarGrafica, nombre_archivo = 'None'):
    """Ajusta un pico gaussiano doble con ODR"""
    parametros, errores, output, gauss_ajustada = ajustar_gaussiana_cuadruple_odr(
        x_data, y_data, x_err, y_err, p0=p0, mostrar_grafica=mostrarGrafica, nombre_archivo = nombre_archivo
    )
    return parametros, errores, output, gauss_ajustada

mostrarGrafica=False,
mostrarGraficaFinal=True,


ruta = "./Experimento V/Datos/"

df_Co_malos = espectro( ruta, 'Co60-18-9.Spe')

df_Cs_malos = espectro( ruta, 'Cs137-18-9-1800s.Spe')

df_Na_malos = espectro( ruta, 'Na22-18-9-1800s.Spe')

df_Ba_malos = espectro( ruta, 'Ba133-18-9-1800s.Spe')

df_Co = df_Co_malos[:900]

df_Cs = df_Cs_malos[:495]

df_Na_malos = df_Na_malos.drop(659)
df_Na = df_Na_malos[:812]

df_Ba = df_Ba_malos[:295]

# --- Definimos arrays base ---
x_Co = df_Co["Canal"].values
y_Co = df_Co["Cuentas"].values

x_Cs = df_Cs["Canal"].values
y_Cs = df_Cs["Cuentas"].values

x_Na = df_Na["Canal"].values
y_Na = df_Na["Cuentas"].values

x_Ba = df_Ba["Canal"].values
y_Ba = df_Ba["Cuentas"].values

x_Co_err = np.full(len(x_Co), 1/2, dtype=float)
y_Co_err = np.sqrt(y_Co)

x_Cs_err = np.full(len(x_Cs), 1/2, dtype=float)
y_Cs_err = np.sqrt(y_Cs)

x_Na_err = np.full(len(x_Na), 1/2, dtype=float)
y_Na_err = np.sqrt(y_Na)

x_Ba_err = np.full(len(x_Ba), 1/2, dtype=float)
y_Ba_err = np.sqrt(y_Ba)

# graficar_con_error(x_Co, y_Co, x_Co_err, y_Co_err, "canal", "cuentas")

# graficar_con_error(x_Cs, y_Cs, x_Cs_err, y_Cs_err, "canal", "cuentas")

# graficar_con_error(x_Na, y_Na, x_Na_err, y_Na_err, "canal", "cuentas")

# graficar_con_error(x_Ba, y_Ba, x_Ba_err, y_Ba_err, "canal", "cuentas")

#PICOS DE AJUSTE LINEAL

#COBALTO 

corte1_Co=[620, 880]
p0_1_Co=[0, 693, 7, 4, 0, 0, 788, 7]

# --- Ajuste del primer pico ---
x1_Co, y1_Co, xerr1_Co, yerr1_Co = cortar_datos(corte1_Co[0], corte1_Co[1], x_Co, y_Co, x_Co_err, y_Co_err)
parametros1_Co, errores1_Co, _, _ = ajustar_pico_gaussiano_doble(x1_Co, y1_Co, xerr1_Co, yerr1_Co, p0_1_Co, mostrarGrafica)

#CESIO

corte1_Cs=[1, 30]
p0_1_Cs=[0, 15, 7, -1, 0]

corte2_Cs=[350, 520]
p0_2_Cs=[0, 400, 7, 4, 0]

# # --- Ajuste del primer pico ---
x1_Cs, y1_Cs, xerr1_Cs, yerr1_Cs = cortar_datos(corte1_Cs[0], corte1_Cs[1], x_Cs, y_Cs, x_Cs_err, y_Cs_err)
parametros1_Cs, errores1_Cs, _, _ = ajustar_pico_gaussiano(x1_Cs, y1_Cs, xerr1_Cs, yerr1_Cs, p0_1_Cs, mostrarGrafica)

# # --- Ajuste del segundo pico ---
x2_Cs, y2_Cs, xerr2_Cs, yerr2_Cs = cortar_datos(corte2_Cs[0], corte2_Cs[1], x_Cs, y_Cs, x_Cs_err, y_Cs_err)
parametros2_Cs, errores2_Cs, _, _ = ajustar_pico_gaussiano(x2_Cs, y2_Cs, xerr2_Cs, yerr2_Cs, p0_2_Cs, mostrarGrafica)


#SODIO

corte1_Na=[230, 360]
p0_1_Na=[0, 304, 7, 4, 0]

corte2_Na=[660, 810]
p0_2_Na=[0, 750, 7, 4, 0]

# --- Ajuste del primer pico ---
x1_Na, y1_Na, xerr1_Na, yerr1_Na = cortar_datos(corte1_Na[0], corte1_Na[1], x_Na, y_Na, x_Na_err, y_Na_err)
parametros1_Na, errores1_Na, _, _ = ajustar_pico_gaussiano(x1_Na, y1_Na, xerr1_Na, yerr1_Na, p0_1_Na, mostrarGrafica)

# --- Ajuste del segundo pico ---
x2_Na, y2_Na, xerr2_Na, yerr2_Na = cortar_datos(corte2_Na[0], corte2_Na[1], x_Na, y_Na, x_Na_err, y_Na_err)
parametros2_Na, errores2_Na, _, _ = ajustar_pico_gaussiano(x2_Na, y2_Na, xerr2_Na, yerr2_Na, p0_2_Na, mostrarGrafica)


#BARIO 

corte1_Ba=[2, 30]
p0_1_Ba=[0, 14, 4, 1, 0]

corte2_Ba=[30, 70]
p0_2_Ba=[0, 47, 7, 4, 0]

corte3_Ba=[150, 250]
p0_3_Ba=[0, 185, 7, 4, 0, 0, 215, 7]


# --- Ajuste del primer pico ---
x1_Ba, y1_Ba, xerr1_Ba, yerr1_Ba = cortar_datos(corte1_Ba[0], corte1_Ba[1], x_Ba, y_Ba, x_Ba_err, y_Ba_err)
parametros1_Ba, errores1_Ba, _, _ = ajustar_pico_gaussiano(x1_Ba, y1_Ba, xerr1_Ba, yerr1_Ba, p0_1_Ba, mostrarGrafica)

# --- Ajuste del segundo pico ---
x2_Ba, y2_Ba, xerr2_Ba, yerr2_Ba = cortar_datos(corte2_Ba[0], corte2_Ba[1], x_Ba, y_Ba, x_Ba_err, y_Ba_err)
parametros2_Ba, errores2_Ba, _, _ = ajustar_pico_gaussiano(x2_Ba, y2_Ba, xerr2_Ba, yerr2_Ba, p0_2_Ba, mostrarGrafica)

# --- Ajuste del tercer pico (doble) ---
x3_Ba, y3_Ba, xerr3_Ba, yerr3_Ba = cortar_datos(corte3_Ba[0], corte3_Ba[1], x_Ba, y_Ba, x_Ba_err, y_Ba_err)
parametros3_Ba, errores3_Ba, _, _ = ajustar_pico_gaussiano_doble(x3_Ba, y3_Ba, xerr3_Ba, yerr3_Ba, p0_3_Ba, mostrarGrafica)


Energia = [1173.2, 1330, 33, 662, 511, 1274, 33, 81, 356.014]
errE_inventado = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
canal = [parametros1_Co[1], parametros1_Co[6], parametros1_Cs[1], parametros2_Cs[1], parametros1_Na[1], parametros2_Na[1], parametros1_Ba[1], parametros2_Ba[1], parametros3_Ba[6]] 
errCanal = [errores1_Co[1], errores1_Co[6], errores1_Cs[1], errores2_Cs[1], errores1_Na[1], errores2_Na[1], errores1_Ba[1], errores2_Ba[1], errores3_Ba[6]]

m, sm, b, sb = fit_lineal(canal, Energia, errCanal, errE_inventado)

E_Co, errE_Co = calibrar(x_Co, x_Co_err, m, b, sm, sb)

E_Cs, errE_Cs = calibrar(x_Cs, x_Cs_err, m, b, sm, sb)

E_Na, errE_Na = calibrar(x_Na, x_Na_err, m, b, sm, sb)

E_Ba, errE_Ba = calibrar(x_Ba, x_Ba_err, m, b, sm, sb)



# def devolver_energia_cuentas(
#     corte1=(13, 60),
#     p0_1=[0, 30, 7, 4, 0],
#     mostrarGrafica=True,
#     corte2=(550, 820),
#     p0_2=[0, 662, 7, 4, 0],
#     mostrarGrafica=True,
#     mostrarGraficaFinal=True,
#     corteRetro = (70, 120),
#     corteCompton = (70, 120),
#     p0_Compton = [0, 320, 8, 2],
#     mostrarGraficaRetro = True, 
#     mostrarGraficaCompton = True, 
#     nombre_archivoRetro = 'Retro',
#     nombre_archivoCompton = 'Compton',
#     ajustarPlomo=False,
#     cortePlomo=(850, 1050),
#     p0_Plomo=None,
#     mostrarGraficaPlomo=True,
#     nombre_archivoPlomo="Plomo"
# ):


#     # --- Calibración ---
#     canal = [parametros1[1], parametros2[1]]
#     errCanal = [errores1[1], errores2[1]]
#     # Esto está mal, porque no sirve un ajuste de dos valores, lo haré a mano
#     # m, sm, b, sb = fit_lineal(canal, Energia, errCanal, errEnergia)

    
#     # print(canal[1] - canal[0])
#     m =(662-32)/(canal[1] - canal[0])
#     b = -m * canal[0] + 32
#     sm = np.sqrt(errCanal[0]**2 + errCanal[1]**2) * ((662-32)/(canal[1] - canal[0])**2)
#     sb = np.sqrt((m * errCanal[0])**2 + (sm * canal[0])**2)
    

#     errorX = np.full(len(df["Canal"][:800]), 1/2, dtype=float)
#     Cuentas = df["Cuentas"][:800]
#     errCuentas = np.sqrt(df["Cuentas"][:800])
#     E, errE = calibrar(df["Canal"][:800], errorX, m, b, sm, sb)
#     # print(f"Errores en E: {errE}")
#     if mostrarGraficaFinal:
#         graficar_con_error(E, Cuentas, errE, errCuentas, 'Energía [keV]', 'Cuentas')
    
#     E_retro, Cuentas_retro, errE_retro, errCuentas_retro = cortar_datos(
#         *corteRetro, E, Cuentas, errE, errCuentas
#     )

#     # --- Estimaciones iniciales ---
#     A0 = np.max(Cuentas_retro) - np.min(Cuentas_retro)
#     mu0 = E_retro[np.argmax(Cuentas_retro)]
#     sigma0 = 10  # ancho estimado (keV)
#     m_lin0 = -2  # pendiente inicial negativa (fondo)
#     b_lin0 = np.min(Cuentas_retro)
#     p0_retro = [A0, mu0, sigma0, m_lin0, b_lin0]

#     # --- Ajuste gaussiano + fondo lineal ---
#     parametros_retro, errores_retro, _, _ = ajustar_pico_gaussiano(
#         E_retro, Cuentas_retro, errE_retro, errCuentas_retro, p0_retro, mostrarGraficaRetro, nombre_archivoRetro
#     )

#     E_Compton, Cuentas_Compton, errE_Compton, errCuentas_Compton = cortar_datos(
#         *corteCompton, E, Cuentas, errE, errCuentas
#     )
#     # --- Ajuste gaussiano + fondo lineal ---
#     parametros_Compton, errores_Compton, _, _ = ajustar_borde_compton(
#         E_Compton, Cuentas_Compton, errE_Compton, errCuentas_Compton, p0_Compton, mostrarGraficaCompton, nombre_archivoCompton
#     )
#     resultados_plomo = None
#     if ajustarPlomo:
#         E_Plomo, Cuentas_Plomo, errE_Plomo, errCuentas_Plomo = cortar_datos(
#             *cortePlomo, E, Cuentas, errE, errCuentas
#         )

#         # Si no se proporcionan parámetros iniciales, estimamos automáticamente
#         if p0_Plomo is None:
#             A0 = np.max(Cuentas_Plomo) - np.min(Cuentas_Plomo)
#             mu0 = E_Plomo[np.argmax(Cuentas_Plomo)]
#             sigma0 = 10
#             m_lin0 = -1
#             b_lin0 = np.min(Cuentas_Plomo)
#             p0_Plomo = [A0, mu0, sigma0, m_lin0, b_lin0]

#         parametros_Plomo, errores_Plomo, _, _ = ajustar_pico_gaussiano(
#             E_Plomo,
#             Cuentas_Plomo,
#             errE_Plomo,
#             errCuentas_Plomo,
#             p0_Plomo,
#             mostrarGraficaPlomo,
#             nombre_archivoPlomo,
#         )

#         resultados_plomo = {
#             "parametros": parametros_Plomo,
#             "errores": errores_Plomo,
#         }
#     # Retornamos resultados
#     return E, errE, errCuentas, {
#         "pico1": {"parametros": parametros1, "errores": errores1},
#         "pico2": {"parametros": parametros2, "errores": errores2},
#         "ajuste_lineal": {"m": m, "sm": sm, "b": b, "sb": sb},
#     }