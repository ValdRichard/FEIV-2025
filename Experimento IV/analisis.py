import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

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
    beta0 = [1.0, 0.0]

    odr = ODR(data, modelo, beta0=beta0)
    out = odr.run()

    # Parámetros ajustados
    m, b = out.beta
    sm, sb = out.sd_beta

    # # Predicción del modelo
    # y_pred = f_lineal(out.beta, x)

    # # Chi² reducido
    # chi2 = np.sum(((y - y_pred) / np.maximum(err_y, 1e-12)) ** 2)
    # dof = len(y) - len(out.beta)
    # chi2_red = chi2 / dof if dof > 0 else np.nan

    # # Coeficiente de determinación R²
    # ss_res = np.sum((y - y_pred) ** 2)
    # ss_tot = np.sum((y - np.mean(y)) ** 2)
    # r2 = 1 - ss_res / ss_tot

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
    fondo = leer_spe(path, 'Fondo-17-9-4700.Spe')
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
                     fmt='o', alpha=0.5, label='Datos', 
                     color='orange', capsize=3)
        
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

def devolver_energia_cuentas(
    df,
    corte1=(13, 60),
    p0_1=[0, 30, 7, 4, 0],
    mostrarGrafica1=True,
    corte2=(550, 820),
    p0_2=[0, 662, 7, 4, 0],
    mostrarGrafica2=True,
    mostrarGraficaFinal=True,
    corteRetro = (70, 120),
    p0_retro = [0, 320, 8, 4, 0],
    corteCompton = (70, 120),
    p0_compton = [0, 320, 8, 4, 0],
    mostrarGraficaRetro = True, 
    mostrarGraficaCompton = True, 
):
    def cortar_espectro(izquierda, derecha, p0, mostrarGrafica):
        x_data = df["Canal"][izquierda:derecha]
        x_err = np.full(len(x_data), 1/1024, dtype=float)
        y_data = df["Cuentas"][izquierda:derecha]
        y_err = np.sqrt(y_data)

        parametros, errores, output, gauss_ajustada = ajustar_gaussiana_odr(
            x_data, y_data, x_err, y_err, p0=p0, mostrar_grafica=mostrarGrafica
        )

        # print("Parámetros ajustados:", parametros)
        # print("Errores estándar:", errores)
        return parametros, errores

    # --- Ajuste de los dos picos ---
    parametros1, errores1 = cortar_espectro(*corte1, p0_1, mostrarGrafica1)
    parametros2, errores2 = cortar_espectro(*corte2, p0_2, mostrarGrafica2)

    # --- Calibración ---
    canal = [parametros1[1], parametros2[1]]
    errCanal = [errores1[1], errores2[1]]
    # Esto está mal, porque no sirve un ajuste de dos valores, lo haré a mano
    # m, sm, b, sb = fit_lineal(canal, Energia, errCanal, errEnergia)

    
    # print(canal[1] - canal[0])
    m =(662-32)/(canal[1] - canal[0])
    b = -m * canal[0] + 32
    sm = np.sqrt(errCanal[0]**2 + errCanal[1]**2) * ((662-32)/(canal[1] - canal[0])**2)
    sb = np.sqrt((m * errCanal[0])**2 + (sm * canal[0])**2)
    

    errorX = np.full(len(df["Canal"][:800]), 1/1024, dtype=float)
    Cuentas = df["Cuentas"][:800]
    errCuentas = np.sqrt(df["Cuentas"][:800])
    E, errE = calibrar(df["Canal"][:800], errorX, m, b, sm, sb)
    # print(f"Errores en E: {errE}")
    if mostrarGraficaFinal:
        graficar_con_error(E, Cuentas, errE, errCuentas, 'Energía (keV)', 'Cuentas')

    
    parametros, errores, output, gauss_ajustada = ajustar_gaussiana_odr(
        E, Cuentas, errCuentas, errE, p0=p0_retro, mostrar_grafica=mostrarGraficaRetro
    )
    
    # Retornamos resultados
    return E, errE, errCuentas, {
        "pico1": {"parametros": parametros1, "errores": errores1},
        "pico2": {"parametros": parametros2, "errores": errores2},
        "ajuste_lineal": {"m": m, "sm": sm, "b": b, "sb": sb},
    }