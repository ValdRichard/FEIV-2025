import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData
from scipy.special import erf

def fit_lineal(x, y, err_x=None, err_y=None, mostrar_grafica=True, label_x = 'Eje X', label_y = 'Eje Y', titulo = 'Title'):
    # Conversión a arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    # Manejo de errores
    if err_x is None:
        err_x = np.full_like(x, 1e-6, dtype=float)
    elif np.isscalar(err_x):
        err_x = np.full_like(x, err_x, dtype=float)

    if err_y is None:
        err_y = np.full_like(y, 1e-6, dtype=float)
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
        
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(titulo)
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

fit_lineal(canal,E,errCanal,errE,False,"Canal","Energpia [keV]","Calibración Am-241 Parte 1")

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

m, sm, b, sb, r2 = fit_lineal(canal,E,errCanal,errE,False,"Canal","Energpia [keV]","Calibración Am-241")

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

        corte1_Ag=[191,210]
        p0_1_Ag=[0,1,80,7.426,0.1]
        x1_Ag, y1_Ag, xerr1_Ag, yerr1_Ag = cortar_datos(corte1_Ag[0], corte1_Ag[1], x, y, x_err, y_err)
        parametros1_Ag, errores1_Ag, _, _ = ajustar_gaussiana_recta_odr(x1_Ag, y1_Ag, xerr1_Ag, yerr1_Ag, p0_1_Ag, False)
        #7.41(1) keV (Ka2 del Ni)

        corte2_Ag=[311,333]
        p0_2_Ag=[0,1,50,11.828,0.1]
        x2_Ag, y2_Ag, xerr2_Ag, yerr2_Ag = cortar_datos(corte2_Ag[0], corte2_Ag[1], x, y, x_err, y_err)
        parametros2_Ag, errores2_Ag, _, _ = ajustar_gaussiana_recta_odr(x2_Ag, y2_Ag, xerr2_Ag, yerr2_Ag, p0_2_Ag, False)
        #11.84(1) keV (Ll del Np)

        corte3_Ag=[446,501]
        p0_3_Ag=[0,1,95,16.913,0.1,220,17.754,0.1]
        x3_Ag, y3_Ag, xerr3_Ag, yerr3_Ag = cortar_datos(corte3_Ag[0], corte3_Ag[1], x, y, x_err, y_err)
        parametros3_Ag, errores3_Ag, _, _ = ajustar_gaussiana_doble_recta_odr(x3_Ag, y3_Ag, xerr3_Ag, yerr3_Ag, p0_3_Ag, False)
        #16.92(1) keV y 17.75(1) keV (Lb4 del Np y Lb1 del Np)

        corte4_Ag=[699,731]
        p0_4_Ag=[0,1,20,26.356,0.1]
        x4_Ag, y4_Ag, xerr4_Ag, yerr4_Ag = cortar_datos(corte4_Ag[0], corte4_Ag[1], x, y, x_err, y_err)
        parametros4_Ag, errores4_Ag, _, _ = ajustar_gaussiana_recta_odr(x4_Ag, y4_Ag, xerr4_Ag, yerr4_Ag, p0_4_Ag, False)
        #26.33(2) keV (Ka1 del Sb)

        #Espectro de Ag, con impurezas de Ni y Sb, y filtraciones de rayos del Np 

    elif j==1:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteK_Co=[179, 214]
        p0_K_Co=[8000,6.915,0.1,8000,7.400,0.1,2000,7.649,0.1]
        xK_Co, yK_Co, xerrK_Co, yerrK_Co = cortar_datos(corteK_Co[0], corteK_Co[1], x, y, x_err, y_err)
        parametrosK_Co, erroresK_Co, _, _ = ajustar_gaussiana_triple_odr(xK_Co, yK_Co, xerrK_Co, yerrK_Co, p0_K_Co, False)
        #6.85(1) keV, 7.42(1) keV y 7.71(1) keV (Ka2 del Co, Ka1 del Ni y Kb1 del Co)

        corteKb_Co=[212,234]
        p0_Kb_Co=[0,1,20,22.164,0.1]
        xKb_Co, yKb_Co, xerrKb_Co, yerrKb_Co = cortar_datos(corteKb_Co[0], corteKb_Co[1], x, y, x_err, y_err)
        parametrosKb_Co, erroresKb_Co, _, _ = ajustar_gaussiana_recta_odr(xKb_Co, yKb_Co, xerrKb_Co, yerrKb_Co, p0_Kb_Co, False)
        #8.14(2) keV (Kb1 del Ni)

        corteKa_Co=[578,625]
        p0_Ka_Co=[0,1,30,22.171,0.1,20,21.801,0.1]
        xKa_Co, yKa_Co, xerrKa_Co, yerrKa_Co = cortar_datos(corteKa_Co[0], corteKa_Co[1], x, y, x_err, y_err)
        parametrosKa_Co, erroresKa_Co, _, _ = ajustar_gaussiana_doble_recta_odr(xKa_Co, yKa_Co, xerrKa_Co, yerrKa_Co, p0_Ka_Co, False)
        #22.20(4) keV y 21.86(5) keV (Ka1 y Ka2 del Ag)

        #Espectro del Co, con impurezas de Ni y Ag

    elif j==2:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")        

        corteKa_Cr=[135, 157]
        p0_Ka_Cr=[0,1,8000,5.4052,0.1]
        xKa_Cr, yKa_Cr, xerrKa_Cr, yerrKa_Cr = cortar_datos(corteKa_Cr[0], corteKa_Cr[1], x, y, x_err, y_err)
        parametrosKa_Cr, erroresKa_Cr, _, _ = ajustar_gaussiana_recta_odr(xKa_Cr, yKa_Cr, xerrKa_Cr, yerrKa_Cr, p0_Ka_Cr, False)
        #5.37(2) keV (Ka2 del Cr)

        corte1_Cr=[182, 236]
        p0_1_Cr=[0,1,250,7.398,0.1,50,8.191,0.1]
        x1_Cr, y1_Cr, xerr1_Cr, yerr1_Cr = cortar_datos(corte1_Cr[0], corte1_Cr[1], x, y, x_err, y_err)
        parametros1_Cr, errores1_Cr, _, _ = ajustar_gaussiana_doble_recta_odr(x1_Cr, y1_Cr, xerr1_Cr, yerr1_Cr, p0_1_Cr, False)
        #7.40(1) keV y 8.19(2) keV (Ka2 del Ni y Kb1 del Ni)

        corte2_Cr=[345, 393]
        p0_2_Cr=[0,1,40,13.379,0.1,20,13.780,0.15,20,13.965,0.1]
        x2_Cr, y2_Cr, xerr2_Cr, yerr2_Cr = cortar_datos(corte2_Cr[0], corte2_Cr[1], x, y, x_err, y_err)
        parametros2_Cr, errores2_Cr, _, _ = ajustar_gaussiana_triple_recta_odr(x2_Cr, y2_Cr, xerr2_Cr, yerr2_Cr, p0_2_Cr, False)
        #13.36(2) keV, 13.73(2) keV y 13.96(3) keV (Lg3 del Pt*, La2 del Np y La1 del Np)

        corte3_Cr=[418,496]
        p0_3_Cr=[0,1,13,16.023,0.1,30,16.794,0.1,15,17.696,0.1]
        x3_Cr, y3_Cr, xerr3_Cr, yerr3_Cr = cortar_datos(corte3_Cr[0], corte3_Cr[1], x, y, x_err, y_err)
        parametros3_Cr, errores3_Cr, _, _ = ajustar_gaussiana_triple_recta_odr(x3_Cr, y3_Cr, xerr3_Cr, yerr3_Cr, p0_3_Cr, False)
        #16.01(2) keV, 16.80(2) keV y 17.74(2) keV (Lb6 del Np, Lb2 del Np y Lb1 del Np)

        corte4_Cr=[583,620]
        p0_4_Cr=[0,1,90,22.088,0.1]
        x4_Cr, y4_Cr, xerr4_Cr, yerr4_Cr = cortar_datos(corte4_Cr[0], corte4_Cr[1], x, y, x_err, y_err)
        parametros4_Cr, errores4_Cr, _, _ = ajustar_gaussiana_recta_odr(x4_Cr, y4_Cr, xerr4_Cr, yerr4_Cr, p0_4_Cr, False)
        #22.12(2) keV (Ka1 del Ag)

        corte5_Cr=[664,733]
        p0_5_Cr=[0,1,20,24.896,0.1,7,25.451,0.1,10,26.391,0.1]
        x5_Cr, y5_Cr, xerr5_Cr, yerr5_Cr = cortar_datos(corte5_Cr[0], corte5_Cr[1], x, y, x_err, y_err)
        parametros5_Cr, errores5_Cr, _, _ = ajustar_gaussiana_triple_recta_odr(x5_Cr, y5_Cr, xerr5_Cr, yerr5_Cr, p0_5_Cr, False)
        #24.86(2) keV, 25.48(3) keV y 26.33(2) keV (Kb1 y Kb2 del Ag y Ka1 del Sb)

        #Espectro del Cr, con impurezas de Ni, Pt, Ag y Sb, y filtraciones de rayos del Np
        

    elif j==3:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKa_Cu=[188,227]
        p0_Ka_Cu=[200,7.400,0.2,1000,7.882,0.1]
        xKa_Cu, yKa_Cu, xerrKa_Cu, yerrKa_Cu = cortar_datos(corteKa_Cu[0], corteKa_Cu[1], x, y, x_err, y_err)
        parametrosKa_Cu, erroresKa_Cu, _, _ = ajustar_gaussiana_doble_odr(xKa_Cu, yKa_Cu, xerrKa_Cu, yerrKa_Cu, p0_Ka_Cu, False)
        #7.31(1) keV y 7.92(1) keV (Ka3 del Ni y Ka1 del Cu)

        corte1_Cu=[231, 245]
        p0_1_Cu=[0,1,30,8.778,0.1]  
        x1_Cu, y1_Cu, xerr1_Cu, yerr1_Cu = cortar_datos(corte1_Cu[0], corte1_Cu[1], x, y, x_err, y_err)
        parametros1_Cu, errores1_Cu, _, _ = ajustar_gaussiana_recta_odr(x1_Cu, y1_Cu, xerr1_Cu, yerr1_Cu, p0_1_Cu, False)
        #8.75(1) keV (Kb1 del Cu)

        corte2_Cu=[342,384]
        p0_2_Cu=[0,1,20,13.287,0.1,10,13.851,0.1]
        x2_Cu, y2_Cu, xerr2_Cu, yerr2_Cu = cortar_datos(corte2_Cu[0], corte2_Cu[1], x, y, x_err, y_err)
        parametros2_Cu, errores2_Cu, _, _ = ajustar_gaussiana_doble_recta_odr(x2_Cu, y2_Cu, xerr2_Cu, yerr2_Cu, p0_2_Cu, False)
        #13.23(3) keV y 13.82(3) keV (Lg2 del Pt* y La2 del Np)

        corte3_Cu=[438,491]
        p0_3_Cu=[0,1,15,16.795,0.1,9,17.611,0.1]
        x3_Cu, y3_Cu, xerr3_Cu, yerr3_Cu = cortar_datos(corte3_Cu[0], corte3_Cu[1], x, y, x_err, y_err)
        parametros3_Cu, errores3_Cu, _, _ = ajustar_gaussiana_doble_recta_odr(x3_Cu, y3_Cu, xerr3_Cu, yerr3_Cu, p0_3_Cu, False)
        #16.77(2) keV y 17.60(3) keV (Lb2 del Np y Lb1 del Np)

        #Espectro del Cu, con impurezas de Ni, Pt y filtraciones de rayos del Np

    elif j==4:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKb_Fe=[182,216]
        p0_Kb_Fe=[0,1,17,6.994,0.1,30,7.417,0.1]
        xKb_Fe, yKb_Fe, xerrKb_Fe, yerrKb_Fe = cortar_datos(corteKb_Fe[0], corteKb_Fe[1], x, y, x_err, y_err)
        parametrosKb_Fe, erroresKb_Fe, _, _ = ajustar_gaussiana_doble_recta_odr(xKb_Fe, yKb_Fe, xerrKb_Fe, yerrKb_Fe, p0_Kb_Fe, False)
        #6.97(2) keV y 7.44(2) keV (Kb1 del Fe y Ka2 del Ni)

        corteKa_Fe=[157,185]
        p0_Ka_Fe=[0,1,65,6.345,0.1]
        xKa_Fe, yKa_Fe, xerrKa_Fe, yerrKa_Fe = cortar_datos(corteKa_Fe[0], corteKa_Fe[1], x, y, x_err, y_err)
        parametrosKa_Fe, erroresKa_Fe, _, _ = ajustar_gaussiana_recta_odr(xKa_Fe, yKa_Fe, xerrKa_Fe, yerrKa_Fe, p0_Ka_Fe, False)
        #6.34(1) keV (Ka1 del Fe)

        #Espectro del Fe, con impurezas de Ni

    elif j==5:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKa_Mn=[143,184]
        p0_Ka_Mn=[0,1,60,5.798,0.1,18,6.423,0.1]
        xKa_Mn, yKa_Mn, xerrKa_Mn, yerrKa_Mn = cortar_datos(corteKa_Mn[0], corteKa_Mn[1], x, y, x_err, y_err)
        parametrosKa_Mn, erroresKa_Mn, _, _ = ajustar_gaussiana_doble_recta_odr(xKa_Mn, yKa_Mn, xerrKa_Mn, yerrKa_Mn, p0_Ka_Mn, False)
        #5.89(1) keV y 6.44(1) (Ka1 y Kb1 del Mn)

        corte1_Mn=[192,215]
        p0_1_Mn=[0,1,30,7.419,0.1]
        x1_Mn, y1_Mn, xerr1_Mn, yerr1_Mn = cortar_datos(corte1_Mn[0], corte1_Mn[1], x, y, x_err, y_err)
        parametros1_Mn, errores1_Mn, _, _ = ajustar_gaussiana_recta_odr(x1_Mn, y1_Mn, xerr1_Mn, yerr1_Mn, p0_1_Mn, False)
        #7.41(1) keV (Ka1 del Ni)

        corte2_Mn=[216,234]
        p0_2_Mn=[0,1,10,8.189,0.1]
        x2_Mn, y2_Mn, xerr2_Mn, yerr2_Mn = cortar_datos(corte2_Mn[0], corte2_Mn[1], x, y, x_err, y_err)
        parametros2_Mn, errores2_Mn, _, _ = ajustar_gaussiana_recta_odr(x2_Mn, y2_Mn, xerr2_Mn, yerr2_Mn, p0_2_Mn, False)
        #8.19(2) keV (Kb1 del Ni)

        corte3_Mn=[590,613]
        p0_3_Mn=[0,1,11,22.094,0.1]
        x3_Mn, y3_Mn, xerr3_Mn, yerr3_Mn = cortar_datos(corte3_Mn[0], corte3_Mn[1], x, y, x_err, y_err)
        parametros3_Mn, errores3_Mn, _, _ = ajustar_gaussiana_recta_odr(x3_Mn, y3_Mn, xerr3_Mn, yerr3_Mn, p0_3_Mn, False)
        #22.10(1) keV (Ka1 del Ag)

        #Espectro del Mn, con impurezas de Ni y Ag


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

        corte1_Mo=[188,236]
        p0_1_Mo=[0,1,70,7.409,0.1,25,8.179,0.1]
        x1_Mo, y1_Mo, xerr1_Mo, yerr1_Mo = cortar_datos(corte1_Mo[0], corte1_Mo[1], x, y, x_err, y_err)
        parametros1_Mo, errores1_Mo, _, _ = ajustar_gaussiana_doble_recta_odr(x1_Mo, y1_Mo, xerr1_Mo, yerr1_Mo, p0_1_Mo, False)
        #7.41(2) keV y 8.17(2) keV (Ka1 y Kb1 del Ni)

        corte2_Mo=[587,613]
        p0_2_Mo=[0,1,40,22.129,0.1]
        x2_Mo, y2_Mo, xerr2_Mo, yerr2_Mo = cortar_datos(corte2_Mo[0], corte2_Mo[1], x, y, x_err, y_err)
        parametros2_Mo, errores2_Mo, _, _ = ajustar_gaussiana_recta_odr(x2_Mo, y2_Mo, xerr2_Mo, yerr2_Mo, p0_2_Mo, False)
        #22.11(2) keV (Ka1 del Ag)

        corte3_Mo=[706,733]
        p0_3_Mo=[0,1,7,26.396,0.15]
        x3_Mo, y3_Mo, xerr3_Mo, yerr3_Mo = cortar_datos(corte3_Mo[0], corte3_Mo[1], x, y, x_err, y_err)
        parametros3_Mo, errores3_Mo, _, _ = ajustar_gaussiana_recta_odr(x3_Mo, y3_Mo, xerr3_Mo, yerr3_Mo, p0_3_Mo, False)
        #26.38(2) keV (Ka1 del Sb)

        #Espectro del Mo, con impurezas de Ni, Ag y Sb

    elif j==7:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKa_Nb=[435,468]
        p0_Ka_Nb=[0,1,140,16.615,0.03]
        xKa_Nb, yKa_Nb, xerrKa_Nb, yerrKa_Nb = cortar_datos(corteKa_Nb[0], corteKa_Nb[1], x, y, x_err, y_err)
        parametrosKa_Nb, erroresKa_Nb, _, _ = ajustar_gaussiana_recta_odr(xKa_Nb, yKa_Nb, xerrKa_Nb, yerrKa_Nb, p0_Ka_Nb, False)
        #16.58(3) keV (Ka1 del Nb)

        corteKb_Nb=[493,520]
        p0_Kb_Nb=[0,1,20,18.625,0.2]
        xKb_Nb, yKb_Nb, xerrKb_Nb, yerrKb_Nb = cortar_datos(corteKb_Nb[0], corteKb_Nb[1], x, y, x_err, y_err)
        parametrosKb_Nb, erroresKb_Nb, _, _ = ajustar_gaussiana_recta_odr(xKb_Nb, yKb_Nb, xerrKb_Nb, yerrKb_Nb, p0_Kb_Nb, False)
        #18.61(2) keV (Kb1 del Nb)

        corte1_Nb=[191,231]
        p0_1_Nb=[0,1,110,7.408,0.1,30,8.190,0.1]
        x1_Nb, y1_Nb, xerr1_Nb, yerr1_Nb = cortar_datos(corte1_Nb[0], corte1_Nb[1], x, y, x_err, y_err)
        parametros1_Nb, errores1_Nb, _, _ = ajustar_gaussiana_doble_recta_odr(x1_Nb, y1_Nb, xerr1_Nb, yerr1_Nb, p0_1_Nb, False)
        #7.41(1) keV y 8.16(2) keV (Ka1 y Kb1 del Ni)

        corte2_Nb=[349,391]
        p0_2_Nb=[0,1,20,13.342,0.1,12,13.925,0.1]
        x2_Nb, y2_Nb, xerr2_Nb, yerr2_Nb = cortar_datos(corte2_Nb[0], corte2_Nb[1], x, y, x_err, y_err)
        parametros2_Nb, errores2_Nb, _, _ = ajustar_gaussiana_doble_recta_odr(x2_Nb, y2_Nb, xerr2_Nb, yerr2_Nb, p0_2_Nb, False)
        #13.30(2) keV y 13.91(3) keV (Ka2 del Rb* y La1 del Np)

        corte3_Nb=[587,613]
        p0_3_Nb=[0,1,40,22.056,0.1]
        x3_Nb, y3_Nb, xerr3_Nb, yerr3_Nb = cortar_datos(corte3_Nb[0], corte3_Nb[1], x, y, x_err, y_err)
        parametros3_Nb, errores3_Nb, _, _ = ajustar_gaussiana_recta_odr(x3_Nb, y3_Nb, xerr3_Nb, yerr3_Nb, p0_3_Nb, False)
        #22.10(1) keV (Ka1 del Ag)

        corte4_Nb=[662,693]
        p0_4_Nb=[0,1,10,24.922,0.15]
        x4_Nb, y4_Nb, xerr4_Nb, yerr4_Nb = cortar_datos(corte4_Nb[0], corte4_Nb[1], x, y, x_err, y_err)
        parametros4_Nb, errores4_Nb, _, _ = ajustar_gaussiana_recta_odr(x4_Nb, y4_Nb, xerr4_Nb, yerr4_Nb, p0_4_Nb, False)
        #24.96(2) keV (Kb1 del Ag)

        corte5_Nb=[704,727]
        p0_5_Nb=[0,1,13,26.356,0.15]
        x5_Nb, y5_Nb, xerr5_Nb, yerr5_Nb = cortar_datos(corte5_Nb[0], corte5_Nb[1], x, y, x_err, y_err)
        parametros5_Nb, errores5_Nb, _, _ = ajustar_gaussiana_recta_odr(x5_Nb, y5_Nb, xerr5_Nb, yerr5_Nb, p0_5_Nb, False)
        #26.31(2) keV (Ka1 del Sb)

        #Espectro del Nb, con impurezas de Ni, Rb*, Ag y Sb, y filtraciones de rayos del Np


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

    elif j==9:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteLg_Pd=[73,107]
        p0_Lg_Pd=[0,1,25,3.328,0.1]
        xLg_Pd, yLg_Pd, xerrLg_Pd, yerrLg_Pd = cortar_datos(corteLg_Pd[0], corteLg_Pd[1], x, y, x_err, y_err)
        parametrosLg_Pd, erroresLg_Pd, _, _ = ajustar_gaussiana_recta_odr(xLg_Pd, yLg_Pd, xerrLg_Pd, yerrLg_Pd, p0_Lg_Pd, False)
        #3.32(3) keV (Lg del Pd)

        corteKa_Pd=[550,624]
        p0_Ka_Pd=[0,1,22,20.746,0.1,75,21.020,0.1,65,22.168,0.1]
        xKa_Pd, yKa_Pd, xerrKa_Pd, yerrKa_Pd = cortar_datos(corteKa_Pd[0], corteKa_Pd[1], x, y, x_err, y_err)
        parametrosKa_Pd, erroresKa_Pd, _, _ = ajustar_gaussiana_triple_recta_odr(xKa_Pd, yKa_Pd, xerrKa_Pd, yerrKa_Pd, p0_Ka_Pd, False)
        #20.63(3) keV, 21.12(2) keV y 22.11(2) keV (Ka3, Ka1 y ??? del Pd)

    elif j==10:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")
    
        corteKa_Ru=[513,535]
        p0_Ka_Ru=[0,1,450,19.279,0.1]
        xKa_Ru, yKa_Ru, xerrKa_Ru, yerrKa_Ru = cortar_datos(corteKa_Ru[0], corteKa_Ru[1], x, y, x_err, y_err)
        parametrosKa_Ru, erroresKa_Ru, _, _ = ajustar_gaussiana_recta_odr(xKa_Ru, yKa_Ru, xerrKa_Ru, yerrKa_Ru, p0_Ka_Ru, False)
        #19.24(1) keV (Ka1 del Ru)

        corteKb_Ru=[580,613]
        p0_Kb_Ru=[0,1,100,21.655,0.1,300,22.070,0.1]
        xKb_Ru, yKb_Ru, xerrKb_Ru, yerrKb_Ru = cortar_datos(corteKb_Ru[0], corteKb_Ru[1], x, y, x_err, y_err)
        parametrosKb_Ru, erroresKb_Ru, _, _ = ajustar_gaussiana_doble_recta_odr(xKb_Ru, yKb_Ru, xerrKb_Ru, yerrKb_Ru, p0_Kb_Ru, False)
        #21.64(1) keV y 22.10(1) keV (Kb1 y Kb2 del Ru)

    elif j==11:
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

    elif j==12:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteLa_Sn=[79,107]
        p0_La_Sn=[0,1,80,3.378,0.1,70,3.626,0.1]
        xLa_Sn, yLa_Sn, xerrLa_Sn, yerrLa_Sn = cortar_datos(corteLa_Sn[0], corteLa_Sn[1], x, y, x_err, y_err)
        parametrosLa_Sn, erroresLa_Sn, _, _ = ajustar_gaussiana_doble_recta_odr(xLa_Sn, yLa_Sn, xerrLa_Sn, yerrLa_Sn, p0_La_Sn, False)
        #3.35(1) keV y 3.71(2) keV (La1 y Lb4 del Sn)

        corteKa_Sn=[674,705]
        p0_Ka_Sn=[0,1,190,25.271,0.03]
        xKa_Sn, yKa_Sn, xerrKa_Sn, yerrKa_Sn = cortar_datos(corteKa_Sn[0], corteKa_Sn[1], x, y, x_err, y_err)
        parametrosKa_Sn, erroresKa_Sn, _, _ = ajustar_gaussiana_recta_odr(xKa_Sn, yKa_Sn, xerrKa_Sn, yerrKa_Sn, p0_Ka_Sn, False)
        #25.21(1) keV (Ka1 del Sn)

    elif j==13:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteLl_W=[196,214]
        p0_Ll_W=[0,1,50,7.387,0.05]
        xLl_W, yLl_W, xerrLl_W, yerrLl_W = cortar_datos(corteLl_W[0], corteLl_W[1], x, y, x_err, y_err)
        parametrosLl_W, erroresLl_W, _, _ = ajustar_gaussiana_recta_odr(xLl_W, yLl_W, xerrLl_W, yerrLl_W, p0_Ll_W, False)
        #7.41(2) keV (Ll del W)

        corteLa_W=[215,239]
        p0_La_W=[0,1,100,8.398,0.15]
        xLa_W, yLa_W, xerrLa_W, yerrLa_W = cortar_datos(corteLa_W[0], corteLa_W[1], x, y, x_err, y_err)
        parametrosLa_W, erroresLa_W, _, _ = ajustar_gaussiana_recta_odr(xLa_W, yLa_W, xerrLa_W, yerrLa_W, p0_La_W, False)
        #8.34(1) keV (La1 del W)

        corteLb_W=[250,279]
        p0_Lb_W=[0,1,90,9.672,0.1]
        xLb_W, yLb_W, xerrLb_W, yerrLb_W = cortar_datos(corteLb_W[0], corteLb_W[1], x, y, x_err, y_err)
        parametrosLb_W, erroresLb_W, _, _ = ajustar_gaussiana_recta_odr(xLb_W, yLb_W, xerrLb_W, yerrLb_W, p0_Lb_W, False)
        #9.67(1) keV (Lb1 del W)

        corteLg_W=[295,319]
        p0_Lg_W=[0,1,9,11.288,0.39]
        xLg_W, yLg_W, xerrLg_W, yerrLg_W = cortar_datos(corteLg_W[0], corteLg_W[1], x, y, x_err, y_err)
        parametrosLg_W, erroresLg_W, _, _ = ajustar_gaussiana_recta_odr(xLg_W, yLg_W, xerrLg_W, yerrLg_W, p0_Lg_W, False)
        #11.28(2) keV (Lg1 del W)

    elif j==14:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKa_Zn=[222, 246]
        p0_Ka_Zn=[0,1,590,8.637,0.1]
        xKa_Zn, yKa_Zn, xerrKa_Zn, yerrKa_Zn = cortar_datos(corteKa_Zn[0], corteKa_Zn[1], x, y, x_err, y_err)
        parametrosKa_Zn, erroresKa_Zn, _, _ = ajustar_gaussiana_recta_odr(xKa_Zn, yKa_Zn, xerrKa_Zn, yerrKa_Zn, p0_Ka_Zn, False)
        #8.59(1) keV (Ka1 del Zn)

        corteKb_Zn=[253, 267]
        p0_Kb_Zn=[0,1,80,9.570,0.1]
        xKb_Zn, yKb_Zn, xerrKb_Zn, yerrKb_Zn = cortar_datos(corteKb_Zn[0], corteKb_Zn[1], x, y, x_err, y_err)
        parametrosKb_Zn, erroresKb_Zn, _, _ = ajustar_gaussiana_recta_odr(xKb_Zn, yKb_Zn, xerrKb_Zn, yerrKb_Zn, p0_Kb_Zn, False)
        #9.53(1) keV (Kb1 del Zn)

    elif j==15:
        #graficar_con_error(x,y,x_err,y_err,"Energía [keV]","Cuentas",titulos[j])
        #graficar(x0,y,"Canales","Cuentas")

        corteKa_Zr=[415, 442]
        p0_Ka_Zr=[0,1,255,15.775,0.1]
        xKa_Zr, yKa_Zr, xerrKa_Zr, yerrKa_Zr = cortar_datos(corteKa_Zr[0], corteKa_Zr[1], x, y, x_err, y_err)
        parametrosKa_Zr, erroresKa_Zr, _, _ = ajustar_gaussiana_recta_odr(xKa_Zr, yKa_Zr, xerrKa_Zr, yerrKa_Zr, p0_Ka_Zr, False)
        #15.74(1) keV (Ka1 del Zr)

        corteKb_Zr=[469, 497]
        p0_Kb_Zr=[0,1,50,17.668,0.1]
        xKb_Zr, yKb_Zr, xerrKb_Zr, yerrKb_Zr = cortar_datos(corteKb_Zr[0], corteKb_Zr[1], x, y, x_err, y_err)
        parametrosKb_Zr, erroresKb_Zr, _, _ = ajustar_gaussiana_recta_odr(xKb_Zr, yKb_Zr, xerrKb_Zr, yerrKb_Zr, p0_Kb_Zr, False)
        #17.70(1) keV (Kb1 del Zr)

def leer_datos_espectros(ruta_archivo):
    df = pd.read_csv(ruta_archivo, sep='\t', engine='python')
    encabezado = df.iloc[1].tolist()
    df = df[2:].reset_index(drop=True)
    df.columns = ['Elemento'] + encabezado[1:]
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(how='all', subset=df.columns[1:])
    return df

df_datos = leer_datos_espectros('Experimento VI/Datos/Datos tpVI - Hoja 1.tsv')

y_Z = df_datos['Z'].values
x_Ka = np.sqrt(df_datos['Ka'].values)
x_Kb = np.sqrt(df_datos['Kb'].values)
x_La = np.sqrt(df_datos['La'].values)
x_Lb = np.sqrt(df_datos['Lb'].values)

yerr_Z = np.full(len(y_Z), 0.1)
xerr_Ka =df_datos['errKa'].values / (2 * x_Ka)
xerr_Kb =df_datos['errKb'].values / (2 * x_Kb)
xerr_La =df_datos['errLa'].values / (2 * x_La)
xerr_Lb =df_datos['errLb'].values / (2 * x_Lb)
 

def limpiar_datos(x, y, xerr, yerr):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(xerr) & np.isfinite(yerr)
    return x[mask], y[mask], xerr[mask], yerr[mask]

x_Ka, y_Ka, xerr_Ka, yerr_Ka = limpiar_datos(x_Ka, y_Z, xerr_Ka, yerr_Z)
x_Kb, y_Kb, xerr_Kb, yerr_Kb = limpiar_datos(x_Kb, y_Z, xerr_Kb, yerr_Z)
x_La, y_La, xerr_La, yerr_La = limpiar_datos(x_La, y_Z, xerr_La, yerr_Z)
x_Lb, y_Lb, xerr_Lb, yerr_Lb = limpiar_datos(x_Lb, y_Z, xerr_Lb, yerr_Z)

#graficar(x_Ka, y_Z, "√Ka (keV^0.5)", "Número atómico Z")
fit_lineal(x_Ka, y_Ka, xerr_Ka, yerr_Ka, False, "√E (keV^0.5)", "Número atómico Z", titulo = "Ajuste de la ley de Moseley para líneas Ka")
fit_lineal(x_Kb, y_Kb, xerr_Kb, yerr_Kb, False, "√E (keV^0.5)", "Número atómico Z", titulo = "Ajuste de la ley de Moseley para líneas Kb")
fit_lineal(x_La, y_La, xerr_La, yerr_La, False, "√E (keV^0.5)", "Número atómico Z", titulo = "Ajuste de la ley de Moseley para líneas La")
fit_lineal(x_Lb, y_Lb, xerr_Lb, yerr_Lb, False, "√E (keV^0.5)", "Número atómico Z", titulo = "Ajuste de la ley de Moseley para líneas Lb")

plt.figure(figsize=(8,5))

# Conjunto 1
plt.errorbar(x_Ka, y_Ka, xerr=xerr_Ka, yerr=yerr_Ka, fmt='o', label='Ka', capsize=3)
plt.plot(x_Ka, np.poly1d(np.polyfit(x_Ka, y_Ka, 1))(x_Ka), '-', color='C0')

# Conjunto 2
plt.errorbar(x_Kb, y_Kb, xerr=xerr_Kb, yerr=yerr_Kb, fmt='o', label='Kb', capsize=3)
plt.plot(x_Kb, np.poly1d(np.polyfit(x_Kb, y_Kb, 1))(x_Kb), '-', color='C1')

# Conjunto 3
plt.errorbar(x_La, y_La, xerr=xerr_La, yerr=yerr_La, fmt='o', label='La', capsize=3)
plt.plot(x_La, np.poly1d(np.polyfit(x_La, y_La, 1))(x_La), '-', color='C2')

# Conjunto 4
plt.errorbar(x_Lb, y_Lb, xerr=xerr_Lb, yerr=yerr_Lb, fmt='o', label='Lb', capsize=3)
plt.plot(x_Lb, np.poly1d(np.polyfit(x_Lb, y_Lb, 1))(x_Lb), '-', color='C3')

plt.xlabel('√E (√keV)')
plt.ylabel('Número atómico Z')
plt.title('Verificación de la Ley De Moseley')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

