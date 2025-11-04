import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.special import erf

def fit_lineal(x, y, err_x=None, err_y=None, mostrar_grafica=True):
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
        
        plt.xlabel('Canal')
        plt.ylabel('Energía [keV]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return m, sm, b, sb, r2

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
    end = 763
    counts = [int(x) for x in lines[start:end] if x]

    df = pd.DataFrame({
        "Canal": range(len(counts)),
        "Cuentas": counts
    })
    return df

def cortar_datos(izquierda, derecha, x, y, x_err, y_err):
    x_data = x[izquierda:derecha]
    y_data = y[izquierda:derecha]
    x_err = x_err[izquierda:derecha]
    y_err = y_err[izquierda:derecha]
    return x_data, y_data, x_err, y_err

def funcion_gaussiana_recta(beta, x):
    """
    Función gaussiana para ODR.
    beta[0] = pendiente
    beta[1] = ordenada
    beta[2] = amplitud
    beta[3] = media
    beta[4] = sigma
    """
    return beta[0] * x + beta[1] + beta[2] * np.exp(-(x - beta[3])**2 / (2 * beta[4]**2))

def funcion_gaussiana(beta, x):
    """
    Función gaussiana para ODR.
    beta[0] = amplitud
    beta[1] = media
    beta[2] = sigma
    """
    return beta[0] * np.exp(-(x - beta[1])**2 / (2 * beta[2]**2))

def funcion_gaussiana_doble_recta(beta, x):
    """
    Función gaussiana doble para ODR
    beta[0] = pendiente
    beta[1] = ordenada
    beta[2] = amplitud 1
    beta[3] = media 1
    beta[4] = sigma 1
    beta[5] = amplitud 2
    beta[6] = media 2
    beta[7] = sigma 2
    """
    return beta[0] * x + beta[1] + beta[2] * np.exp(-(x - beta[3])**2 / (2 * beta[4]**2)) + beta[5] * np.exp(-(x - beta[6])**2 / (2 * beta[7]**2))

def funcion_gaussiana_doble(beta, x):
    """
    Función gaussiana doble para ODR
    beta[0] = amplitud 1
    beta[1] = media 1
    beta[2] = sigma 1
    beta[3] = amplitud 2
    beta[4] = media 2
    beta[5] = sigma 2
    """
    return beta[0] * np.exp(-(x - beta[1])**2 / (2 * beta[2]**2)) + beta[3] * np.exp(-(x - beta[4])**2 / (2 * beta[5]**2))

def funcion_gaussiana_triple(beta, x):
    """
    Función gaussiana para ODR.
    beta[0] = amplitud1
    beta[1] = media1
    beta[2] = sigma1
    beta[3] = amplitud2
    beta[4] = media2
    beta[5] = sigma2
    beta[6] = amplitud3
    beta[7] = media3
    beta[8] = sigma3
    """
    return beta[0] * np.exp(-(x - beta[1])**2 / (2 * beta[2]**2)) + beta[3] * np.exp(-(x - beta[4])**2 / (2 * beta[5]**2)) + beta[6] * np.exp(-(x - beta[7])**2 / (2 * beta[8]**2))

def funcion_gaussiana_triple_recta(beta, x):
    return beta[0] * x + beta[1] + beta[2] * np.exp(-(x - beta[3])**2 / (2 * beta[4]**2)) + beta[5] * np.exp(-(x - beta[6])**2 / (2 * beta[7]**2)) + beta[8] * np.exp(-(x - beta[9])**2 / (2 * beta[10]**2))

def funcion_gaussiana_triple(beta, x):
    return beta[0] * np.exp(-(x - beta[1])**2 / (2 * beta[2]**2)) + beta[3] * np.exp(-(x - beta[4])**2 / (2 * beta[5]**2)) + beta[6] * np.exp(-(x - beta[7])**2 / (2 * beta[8]**2))

def funcion_gaussiana_cuadruple(beta, x):
    return beta[0] * np.exp(-(x - beta[1])**2 / (2 * beta[2]**2)) + beta[3] * np.exp(-(x - beta[4])**2 / (2 * beta[5]**2)) + beta[6] * np.exp(-(x - beta[7])**2 / (2 * beta[8]**2))  + beta[9] * np.exp(-(x - beta[10])**2 / (2 * beta[11]**2))

def ajustar_gaussiana_recta_odr(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True,
                          nombre_archivo="Gaussiana"):
    if p0 is None:
        p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
    
    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)
    
    modelo_gauss = Model(funcion_gaussiana_recta)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    output = odr.run()
    
    parametros = output.beta
    errores = output.sd_beta
    
    def gaussiana_ajustada(x):
        return funcion_gaussiana_recta(parametros, x)
    
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
                        f'A={parametros[2]:.2f}±{errores[2]:.2f}\n'
                        f'E={parametros[3]:.2f}±{errores[3]:.2f}\n'
                        f'σ={parametros[4]:.2f}±{errores[4]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Energía [keV]')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento VI/Imagenes/Gaussiana"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)

        plt.show()
    
    return parametros, errores, output, gaussiana_ajustada

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
        carpeta = "./Experimento VI/Imagenes/Gaussiana"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)

        plt.show()
    
    return parametros, errores, output, gaussiana_ajustada

def ajustar_gaussiana_doble_recta_odr(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True,
                          nombre_archivo="Gaussiana doble"):
    if p0 is None:
        p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
    
    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)
    
    modelo_gauss = Model(funcion_gaussiana_doble_recta)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    output = odr.run()
    
    parametros = output.beta
    errores = output.sd_beta
    
    def gaussiana_ajustada(x):
        return funcion_gaussiana_doble_recta(parametros, x)
    
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
                        f'A_1={parametros[2]:.2f}±{errores[2]:.2f}\n'
                        f'C_1={parametros[3]:.2f}±{errores[3]:.2f}\n'
                        f'σ_1={parametros[4]:.2f}±{errores[4]:.2f}\n'
                        f'A_2={parametros[5]:.2f}±{errores[5]:.2f}\n'
                        f'C_2={parametros[6]:.2f}±{errores[6]:.2f}\n'
                        f'σ_2={parametros[7]:.2f}±{errores[7]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Canal')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento VI/Imagenes/Gaussiana doble"
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
                        f'A_2={parametros[3]:.2f}±{errores[3]:.2f}\n'
                        f'C_2={parametros[4]:.2f}±{errores[4]:.2f}\n'
                        f'σ_2={parametros[5]:.2f}±{errores[5]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Canal')
        plt.ylabel('Energía [keV]')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento VI/Imagenes/Gaussiana doble"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)

        plt.show()
    
    return parametros, errores, output, gaussiana_ajustada

def ajustar_gaussiana_triple_odr(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True,
                          nombre_archivo="Gaussiana triple"):
    if p0 is None:
        p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
    
    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)
    
    modelo_gauss = Model(funcion_gaussiana_triple)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    output = odr.run()
    
    parametros = output.beta
    errores = output.sd_beta
    
    def gaussiana_ajustada(x):
        return funcion_gaussiana_triple(parametros, x)
    
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
                        f'A_2={parametros[3]:.2f}±{errores[3]:.2f}\n'
                        f'C_2={parametros[4]:.2f}±{errores[4]:.2f}\n'
                        f'σ_2={parametros[5]:.2f}±{errores[5]:.2f}\n'
                        f'A_3={parametros[6]:.2f}±{errores[6]:.2f}\n'
                        f'C_3={parametros[7]:.2f}±{errores[7]:.2f}\n'
                        f'σ_3={parametros[8]:.2f}±{errores[8]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Canal')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento VI/Imagenes/Gaussiana triple"
        os.makedirs(carpeta, exist_ok=True)
        ruta_archivo = f"{carpeta}/{nombre_archivo}.png"
        plt.savefig(ruta_archivo, dpi=300)

        plt.show()
    
    return parametros, errores, output, gaussiana_ajustada

def ajustar_gaussiana_triple_recta_odr(x_data, y_data, 
                          x_err=None, y_err=None, 
                          p0=None, mostrar_grafica=True,
                          nombre_archivo="Gaussiana triple"):
    if p0 is None:
        p0 = [np.max(y_data), np.mean(x_data), np.std(x_data)]
    
    if x_err is None:
        x_err = np.ones_like(x_data) * 0.01 * np.ptp(x_data)
    if y_err is None:
        y_err = np.ones_like(y_data) * 0.01 * np.ptp(y_data)
    
    modelo_gauss = Model(funcion_gaussiana_triple_recta)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    output = odr.run()
    
    parametros = output.beta
    errores = output.sd_beta
    
    def gaussiana_ajustada(x):
        return funcion_gaussiana_triple_recta(parametros, x)
    
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
                        f'A_1={parametros[2]:.2f}±{errores[2]:.2f}\n'
                        f'C_1={parametros[3]:.2f}±{errores[3]:.2f}\n'
                        f'σ_1={parametros[4]:.2f}±{errores[4]:.2f}\n'
                        f'A_2={parametros[5]:.2f}±{errores[5]:.2f}\n'
                        f'C_2={parametros[6]:.2f}±{errores[6]:.2f}\n'
                        f'σ_2={parametros[7]:.2f}±{errores[7]:.2f}\n'
                        f'A_3={parametros[8]:.2f}±{errores[8]:.2f}\n'
                        f'C_3={parametros[9]:.2f}±{errores[9]:.2f}\n'
                        f'σ_3={parametros[10]:.2f}±{errores[10]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Canal')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento VI/Imagenes/Gaussiana triple"
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
    
    modelo_gauss = Model(funcion_gaussiana_cuadruple)
    datos_odr = RealData(x_data, y_data, sx=x_err, sy=y_err)
    odr = ODR(datos_odr, modelo_gauss, beta0=p0)
    output = odr.run()
    
    parametros = output.beta
    errores = output.sd_beta
    
    def gaussiana_ajustada(x):
        return funcion_gaussiana_cuadruple(parametros, x)
    
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
                        f'A_2={parametros[3]:.2f}±{errores[3]:.2f}\n'
                        f'C_2={parametros[4]:.2f}±{errores[4]:.2f}\n'
                        f'σ_2={parametros[5]:.2f}±{errores[5]:.2f}\n'
                        f'A_3={parametros[6]:.2f}±{errores[6]:.2f}\n'
                        f'C_3={parametros[7]:.2f}±{errores[7]:.2f}\n'
                        f'σ_3={parametros[8]:.2f}±{errores[8]:.2f}\n'
                        f'A_4={parametros[9]:.2f}±{errores[9]:.2f}\n'
                        f'C_4={parametros[10]:.2f}±{errores[10]:.2f}\n'
                        f'σ_4={parametros[11]:.2f}±{errores[11]:.2f}\n'
                        f'R²={r2:.4f}'))
        
        plt.xlabel('Canal')
        plt.ylabel('Cuentas')
        plt.legend()
        plt.grid(alpha=0.3)

        # 💾 Guardar imagen
        carpeta = "./Experimento VI/Imagenes/Gaussiana cuadruple"
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

ruta = "./Experimento VI/Datos/"

#Calibración
# --- Creamos los data frames --- 
df_Am = leer_spe( ruta, 'Am241_4_11_2024.Spe') 
x_Am = df_Am["Canal"].values
y_Am = df_Am["Cuentas"].values

x_Am_err = np.full(len(x_Am), 1/2, dtype=float)
y_Am_err = np.sqrt(y_Am)
y_Am_err[y_Am_err == 0] = 0.0001


# --- Graficamos sin y con errores ---
#graficar(x_Am, y_Am, "Canal", "Cuentas")
#graficar_con_error(x_Am, y_Am, x_Am_err, y_Am_err, "Canales", "Cuentas")

#La
corteLa_Am=[367, 394]
p0_La_Am=[0,1,30,379,3]
xLa_Am, yLa_Am, xerrLa_Am, yerrLa_Am = cortar_datos(corteLa_Am[0], corteLa_Am[1], x_Am, y_Am, x_Am_err, y_Am_err)
parametrosLa_Am, erroresLa_Am, _, _ = ajustar_gaussiana_recta_odr(xLa_Am, yLa_Am, xerrLa_Am, yerrLa_Am, p0_La_Am, False)

#Lb1 y Lb2
corteLb_Am=[445, 500]
p0_Lb_Am=[0,459,3,30,481,3]
xLb_Am, yLb_Am, xerrLb_Am, yerrLb_Am = cortar_datos(corteLb_Am[0], corteLb_Am[1], x_Am, y_Am, x_Am_err, y_Am_err)
parametrosLb_Am, erroresLb_Am, _, _ = ajustar_gaussiana_doble_odr(xLb_Am, yLb_Am, xerrLb_Am, yerrLb_Am, p0_Lb_Am, False)

#Ma
corteMa_Am=[60, 120]
p0_Ma_Am=[0,1,0,91,3]
xMa_Am, yMa_Am, xerrMa_Am, yerrMa_Am = cortar_datos(corteMa_Am[0], corteMa_Am[1], x_Am, y_Am, x_Am_err, y_Am_err)
parametrosMa_Am, erroresMa_Am, _, _ = ajustar_gaussiana_recta_odr(xMa_Am, yMa_Am, xerrMa_Am, yerrMa_Am, p0_Ma_Am, False)

#Lg
corteLg_Am=[553, 580]
p0_Lg_Am=[0,1,24,564,3]
xLg_Am, yLg_Am, xerrLg_Am, yerrLg_Am = cortar_datos(corteLg_Am[0], corteLg_Am[1], x_Am, y_Am, x_Am_err, y_Am_err)
parametrosLg_Am, erroresLg_Am, _, _ = ajustar_gaussiana_recta_odr(xLg_Am, yLg_Am, xerrLg_Am, yerrLg_Am, p0_Lg_Am, False)

E=[3.250,13.946,16.794,17.751,20.784]
errE=[0.001,0.001,0.001,0.001,0.001]
canal=[parametrosMa_Am[3],parametrosLa_Am[3],parametrosLb_Am[1],parametrosLb_Am[4],parametrosLg_Am[3]]
errCanal=[erroresMa_Am[3],erroresLa_Am[3],erroresLb_Am[1],erroresLb_Am[4],erroresLg_Am[3]]

fit_lineal(canal,E,errCanal,errE,False)

#Ll
corteLl_Am=[308, 337]
p0_Ll_Am=[0,1,5,322,3]
xLl_Am, yLl_Am, xerrLl_Am, yerrLl_Am = cortar_datos(corteLl_Am[0], corteLl_Am[1], x_Am, y_Am, x_Am_err, y_Am_err)
parametrosLl_Am, erroresLl_Am, _, _ = ajustar_gaussiana_recta_odr(xLl_Am, yLl_Am, xerrLl_Am, yerrLl_Am, p0_Ll_Am, False)

#Ma y Mg
corteM_Am=[73,113]
p0_M_Am=[16,90,0.2,14,102,0.3]
xM_Am, yM_Am, xerrM_Am, yerrM_Am = cortar_datos(corteM_Am[0], corteM_Am[1], x_Am, y_Am, x_Am_err, y_Am_err)
parametrosM_Am, erroresM_Am, _, _ = ajustar_gaussiana_doble_odr(xM_Am, yM_Am, xerrM_Am, yerrM_Am, p0_M_Am, False)

E=[3.250,3.664,11.871,13.946,16.794,17.751,20.784]
errE=[0.001,0.001,0.001,0.001,0.001,0.001,0.001]
canal=[parametrosM_Am[1],parametrosM_Am[4],parametrosLl_Am[3],parametrosLa_Am[3],parametrosLb_Am[1],parametrosLb_Am[4],parametrosLg_Am[3]]
errCanal=[erroresM_Am[1],erroresM_Am[4],erroresLl_Am[3],erroresLa_Am[3],erroresLb_Am[1],erroresLb_Am[4],erroresLg_Am[3]]

m, sm, b, sb, r2 = fit_lineal(canal,E,errCanal,errE,False)

df_Am["Energía"] = df_Am["Canal"] * m + b
x_Am_calibrado = df_Am["Energía"]
x_Am_calibrado_err = np.sqrt(x_Am_calibrado**2 * sm**2 + m**2 * x_Am_err**2 + sb**2 )

#graficar_con_error(x_Am_calibrado, y_Am, x_Am_calibrado_err, y_Am_err, "Energía [keV]", "Cuentas")

#Análisis
archivos = ["Ag.spe","Co.spe","Cr.spe","Cu.spe","Fe.spe","Mn.spe","Mo.spe","Nb.spe","Pb.spe","Pd.spe","Ru.spe","Se.spe","Sn.spe","W.spe","Zn.spe","Zr.spe"]
titulos = ["Ag","Co","Cr","Cu","Fe","Mn","Mo","Nb","Pb","Pd","Ru","Se","Sn","W","Zn","Zr"]
for j,i in enumerate(archivos) : 
    df = leer_spe( ruta, i) 
    df["Energía"] = df["Canal"] * m + b
    x0 = df["Canal"].values
    x = df["Energía"].values
    y = df["Cuentas"].values

    x0_err = np.full(len(x), 1/2, dtype=float)
    x_err = np.sqrt( (m * x0_err)**2 + (x0 * sm)**2 + sb**2 )
    y_err = np.sqrt(y)
    y_err[y_err == 0] = 0.0001

    if j==0:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")
        
        corteLa_Ag=[362, 395]
        p0_La_Ag=[0,1,270,13.909,0.2]
        xLa_Ag, yLa_Ag, xerrLa_Ag, yerrLa_Ag = cortar_datos(corteLa_Ag[0], corteLa_Ag[1], x, y, x_err, y_err)
        parametrosLa_Ag, erroresLa_Ag, _, _ = ajustar_gaussiana_recta_odr(xLa_Ag, yLa_Ag, xerrLa_Ag, yerrLa_Ag, p0_La_Ag, False)
        #13.91(1) keV (La1 del Np)
    
        corteKa_Ag=[582,619]
        p0_Ka_Ag=[0,1,760,22.163,0.2]
        xKa_Ag, yKa_Ag, xerrKa_Ag, yerrKa_Ag = cortar_datos(corteKa_Ag[0], corteKa_Ag[1], x, y, x_err, y_err)
        parametrosKa_Ag, erroresKa_Ag, _, _ = ajustar_gaussiana_recta_odr(xKa_Ag, yKa_Ag, xerrKa_Ag, yerrKa_Ag, p0_Ka_Ag, False)
        #22.12(2) keV (Ka1 del Ag)

        corteLb_Ag=[71,104]
        p0_Lb_Ag=[0,1,50,3.255,0.2]
        xLb_Ag, yLb_Ag, xerrLb_Ag, yerrLb_Ag = cortar_datos(corteLb_Ag[0], corteLb_Ag[1], x, y, x_err, y_err)
        parametrosLb_Ag, erroresLb_Ag, _, _ = ajustar_gaussiana_recta_odr(xLb_Ag, yLb_Ag, xerrLb_Ag, yerrLb_Ag, p0_Lb_Ag, False)
        #3.22(2) keV (Lb3 del Ag)

        corteKb_Ag=[666,700]
        p0_Kb_Ag=[200,24.941,0.2,50,25.4503,0.1]
        xKb_Ag, yKb_Ag, xerrKb_Ag, yerrKb_Ag = cortar_datos(corteKb_Ag[0], corteKb_Ag[1], x, y, x_err, y_err)
        parametrosKb_Ag, erroresKb_Ag, _, _ = ajustar_gaussiana_doble_odr(xKb_Ag, yKb_Ag, xerrKb_Ag, yerrKb_Ag, p0_Kb_Ag, False)
        #24.97(2) keV y 25.52(8) keV (Kb1 y Kb2 del Ag)

    elif j==1:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteK_Co=[179, 214]
        p0_K_Co=[8000,6.915,0.1,8000,7.400,0.1,2000,7.649,0.1]
        xK_Co, yK_Co, xerrK_Co, yerrK_Co = cortar_datos(corteK_Co[0], corteK_Co[1], x, y, x_err, y_err)
        parametrosK_Co, erroresK_Co, _, _ = ajustar_gaussiana_triple_odr(xK_Co, yK_Co, xerrK_Co, yerrK_Co, p0_K_Co, False)
        #6.85(1) keV, 7.42(1) keV y 7.71(1) keV (??? del Co)

        corteKb_Co=[212,234]
        p0_Kb_Co=[0,1,20,8.155,0.1]
        xKb_Co, yKb_Co, xerrKb_Co, yerrKb_Co = cortar_datos(corteKb_Co[0], corteKb_Co[1], x, y, x_err, y_err)
        parametrosKb_Co, erroresKb_Co, _, _ = ajustar_gaussiana_recta_odr(xKb_Co, yKb_Co, xerrKb_Co, yerrKb_Co, p0_Kb_Co, False)
        #8.14(2) keV (??? del Co)

    elif j==2:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")        

        corteKa_Cr=[135, 157]
        p0_Ka_Cr=[0,1,8000,5.4052,0.1]
        xKa_Cr, yKa_Cr, xerrKa_Cr, yerrKa_Cr = cortar_datos(corteKa_Cr[0], corteKa_Cr[1], x, y, x_err, y_err)
        parametrosKa_Cr, erroresKa_Cr, _, _ = ajustar_gaussiana_recta_odr(xKa_Cr, yKa_Cr, xerrKa_Cr, yerrKa_Cr, p0_Ka_Cr, False)
        #5.37(2) keV (Ka2 del Cr)

    elif j==3:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")   

        corteKa_Cu=[188,227]
        p0_Ka_Cu=[200,7.400,0.2,1000,7.882,0.1]
        xKa_Cu, yKa_Cu, xerrKa_Cu, yerrKa_Cu = cortar_datos(corteKa_Cu[0], corteKa_Cu[1], x, y, x_err, y_err)
        parametrosKa_Cu, erroresKa_Cu, _, _ = ajustar_gaussiana_doble_odr(xKa_Cu, yKa_Cu, xerrKa_Cu, yerrKa_Cu, p0_Ka_Cu, False)
        #7.92(1) keV (Ka3 del Cu)  

    elif j==4:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKb_Fe=[182,216]
        p0_Kb_Fe=[0,1,17,6.994,0.1,30,7.417,0.1]
        xKb_Fe, yKb_Fe, xerrKb_Fe, yerrKb_Fe = cortar_datos(corteKb_Fe[0], corteKb_Fe[1], x, y, x_err, y_err)
        parametrosKb_Fe, erroresKb_Fe, _, _ = ajustar_gaussiana_doble_recta_odr(xKb_Fe, yKb_Fe, xerrKb_Fe, yerrKb_Fe, p0_Kb_Fe, False)
        #6.97(2) keV y 7.44(2) keV (??? del Fe)

        corteKa_Fe=[157,185]
        p0_Ka_Fe=[0,1,65,6.345,0.1]
        xKa_Fe, yKa_Fe, xerrKa_Fe, yerrKa_Fe = cortar_datos(corteKa_Fe[0], corteKa_Fe[1], x, y, x_err, y_err)
        parametrosKa_Fe, erroresKa_Fe, _, _ = ajustar_gaussiana_recta_odr(xKa_Fe, yKa_Fe, xerrKa_Fe, yerrKa_Fe, p0_Ka_Fe, False)
        #6.34(1) keV (Ka1 del Fe)

    elif j==5:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKa_Mn=[143,184]
        p0_Ka_Mn=[0,1,60,5.798,0.1,18,6.423,0.1]
        xKa_Mn, yKa_Mn, xerrKa_Mn, yerrKa_Mn = cortar_datos(corteKa_Mn[0], corteKa_Mn[1], x, y, x_err, y_err)
        parametrosKa_Mn, erroresKa_Mn, _, _ = ajustar_gaussiana_doble_recta_odr(xKa_Mn, yKa_Mn, xerrKa_Mn, yerrKa_Mn, p0_Ka_Mn, False)
        #5.89(1) keV y 6.44(!) (Ka1 y ??? del Mn)

    elif j==6:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKa_Mo=[457,491]
        p0_Ka_Mo=[0,1,90,17.440,0.03]
        xKa_Mo, yKa_Mo, xerrKa_Mo, yerrKa_Mo = cortar_datos(corteKa_Mo[0], corteKa_Mo[1], x, y, x_err, y_err)
        parametrosKa_Mo, erroresKa_Mo, _, _ = ajustar_gaussiana_recta_odr(xKa_Mo, yKa_Mo, xerrKa_Mo, yerrKa_Mo, p0_Ka_Mo, False)
        #17.46(2) keV (Ka1 del Mo)

        corteKb_Mo=[515,562]
        p0_Kb_Mo=[0,3,5,19.606,0.2]
        xKb_Mo, yKb_Mo, xerrKb_Mo, yerrKb_Mo = cortar_datos(corteKb_Mo[0], corteKb_Mo[1], x, y, x_err, y_err)
        parametrosKb_Mo, erroresKb_Mo, _, _ = ajustar_gaussiana_recta_odr(xKb_Mo, yKb_Mo, xerrKb_Mo, yerrKb_Mo, p0_Kb_Mo, False)
        #19.66(3) keV (Kb1 del Mo)

    elif j==8:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteLa_Pb=[273, 299]
        p0_La_Pb=[0,1,2200,10.551,0.05]
        xLa_Pb, yLa_Pb, xerrLa_Pb, yerrLa_Pb = cortar_datos(corteLa_Pb[0], corteLa_Pb[1], x, y, x_err, y_err)
        parametrosLa_Pb, erroresLa_Pb, _, _ = ajustar_gaussiana_recta_odr(xLa_Pb, yLa_Pb, xerrLa_Pb, yerrLa_Pb, p0_La_Pb, False)
        #10.50(2) keV (La1 del Pb)
        
        corteLb_Pb=[325, 361]
        p0_Lb_Pb=[0,1,1300,12.6,0.05]
        xLb_Pb, yLb_Pb, xerrLb_Pb, yerrLb_Pb = cortar_datos(corteLb_Pb[0], corteLb_Pb[1], x, y, x_err, y_err)
        parametrosLb_Pb, erroresLb_Pb, _, _ = ajustar_gaussiana_recta_odr(xLb_Pb, yLb_Pb, xerrLb_Pb, yerrLb_Pb, p0_Lb_Pb, False)
        #12.59(3) keV (Lb1 del Pb)
        
        corteLl_Pb=[238, 259]
        p0_Ll_Pb=[0,1,1300,9.184,0.05]
        xLl_Pb, yLl_Pb, xerrLl_Pb, yerrLl_Pb = cortar_datos(corteLl_Pb[0], corteLl_Pb[1], x, y, x_err, y_err)
        parametrosLl_Pb, erroresLl_Pb, _, _ = ajustar_gaussiana_recta_odr(xLl_Pb, yLl_Pb, xerrLl_Pb, yerrLl_Pb, p0_Ll_Pb, False)
        #9.14(1) keV (Ll del Pb)

        corteLg_Pb=[391, 409]
        p0_Lg_Pb=[0,1,1300,14.766,0.05]
        xLg_Pb, yLg_Pb, xerrLg_Pb, yerrLg_Pb = cortar_datos(corteLg_Pb[0], corteLg_Pb[1], x, y, x_err, y_err)
        parametrosLg_Pb, erroresLg_Pb, _, _ = ajustar_gaussiana_recta_odr(xLg_Pb, yLg_Pb, xerrLg_Pb, yerrLg_Pb, p0_Lg_Pb, False)
        #14.74(1) keV (Lg1 del Pb)
        
        corteMa_Pb=[55, 73]
        p0_Ma_Pb=[0,1,500,2.342,0.1]
        xMa_Pb, yMa_Pb, xerrMa_Pb, yerrMa_Pb = cortar_datos(corteMa_Pb[0], corteMa_Pb[1], x, y, x_err, y_err)
        parametrosMa_Pb, erroresMa_Pb, _, _ = ajustar_gaussiana_recta_odr(xMa_Pb, yMa_Pb, xerrMa_Pb, yerrMa_Pb, p0_Ma_Pb, False)
        #2.30(1) keV (Ma del Pb)

    if j==11:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")
    
        corteKa_Se=[292, 318]
        p0_Ka_Se=[0,1,400,11.224,0.1]
        xKa_Se, yKa_Se, xerrKa_Se, yerrKa_Se = cortar_datos(corteKa_Se[0], corteKa_Se[1], x, y, x_err, y_err)
        parametrosKa_Se, erroresKa_Se, _, _ = ajustar_gaussiana_recta_odr(xKa_Se, yKa_Se, xerrKa_Se, yerrKa_Se, p0_Ka_Se, False)
        #11.20(1) keV (La1 del Se)

        corteKb_Se=[333, 349]
        p0_Kb_Se=[0,1,200,12.497,0.1]
        xKb_Se, yKb_Se, xerrKb_Se, yerrKb_Se = cortar_datos(corteKb_Se[0], corteKb_Se[1], x, y, x_err, y_err)
        parametrosKb_Se, erroresKb_Se, _, _ = ajustar_gaussiana_recta_odr(xKb_Se, yKb_Se, xerrKb_Se, yerrKb_Se, p0_Kb_Se, False)
        #12.47(2) keV (Lb1 del Se)