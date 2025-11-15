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

ruta = "./Experimento VI/Datos/"

df_Am = leer_spe( ruta, 'Am241.Spe') 
x_Am = df_Am["Canal"].values
y_Am = df_Am["Cuentas"].values

x_Am_err = np.full(len(x_Am), 1/2, dtype=float)
y_Am_err = np.sqrt(y_Am)
y_Am_err[y_Am_err == 0] = 0.0001


#graficar(x_Am_2, y_Am_2, "Canal", "Cuentas")

#La1
corteLa_Am=[368, 390]
p0_La_Am=[0,1,200,379,3]
xLa_Am, yLa_Am, xerrLa_Am, yerrLa_Am = cortar_datos(corteLa_Am[0], corteLa_Am[1], x_Am, y_Am, x_Am_err, y_Am_err)
parametrosLa_Am, erroresLa_Am, _, _ = ajustar_gaussiana_recta_odr(xLa_Am, yLa_Am, xerrLa_Am, yerrLa_Am, p0_La_Am, False)
#379.2(1)

#Lb1
corteLb_Am=[471, 496]
p0_Lb_Am=[0,1,120,483,3]
xLb_Am, yLb_Am, xerrLb_Am, yerrLb_Am = cortar_datos(corteLb_Am[0], corteLb_Am[1], x_Am, y_Am, x_Am_err, y_Am_err)
parametrosLb_Am, erroresLb_Am, _, _ = ajustar_gaussiana_recta_odr(xLb_Am, yLb_Am, xerrLb_Am, yerrLb_Am, p0_Lb_Am, False)
#483.1(2)

m0=0.037
b0=-1.612
sm0=0.01
sb0=0.001

df_A = leer_spe( ruta, 'anillo2.Spe') 
df_A["Energía"] = df_A["Canal"] * m0 + b0
x0_A = df_A["Canal"].values
x_A = df_A["Energía"].values
y_A = df_A["Cuentas"].values

x0_err_A = np.full(len(x_A), 1/2, dtype=float)
x_err_A = np.sqrt( (m0 * x0_err_A)**2 + (x0_A * sm0)**2 + sb0**2 )
y_err_A = np.sqrt(y_A)
y_err_A[y_err_A == 0] = 0.0001

#graficar_con_error(x_A, y_A, x_err_A, y_err_A, "Energía [keV]", "Cuentas")
#graficar(x0_A   , y_A, "Canal", "Cuentas")
