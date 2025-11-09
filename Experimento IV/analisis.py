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



def graficar_con_error(x, y, xerr, yerr, xlabel, ylabel, titulo=None, regiones=None):
    """
    Grafica datos con barras de error y permite marcar regiones y picos (fotopicos).

    Los fotopicos se muestran como líneas punteadas con etiquetas al costado.
    Las regiones muestran su etiqueta centrada, con alpha de texto independiente.
    """

    plt.figure(figsize=(10, 6))

    # Datos experimentales
    plt.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        fmt='.', color='orange',
        ecolor='gray', elinewidth=1, capsize=3,
        alpha=0.5, label='Datos experimentales'
    )

    # Marcar regiones y picos
    if regiones:
        for reg in regiones:
            tipo = reg.get('tipo', 'region')

            # === REGIONES ===
            if tipo == 'region':
                plt.axvspan(
                    reg['xmin'], reg['xmax'],
                    color=reg.get('color', 'lightblue'),
                    alpha=reg.get('alpha', 0.3)
                )

                if reg.get('label'):
                    plt.text(
                        (reg['xmin'] + reg['xmax']) / 2,
                        max(y) * reg.get('y_pos', 0.4),
                        reg['label'],
                        ha='center',
                        va='bottom',
                        fontsize=15,
                        color=reg.get('color_texto', 'black'),
                        alpha=reg.get('alpha_text', 0.95),  # texto más visible
                        fontweight='bold'
                    )

            # === PICOS / FOTOPICOS ===
            elif tipo in ['pico', 'fotopico']:
                plt.axvline(
                    reg['x'],
                    color=reg.get('color', 'black'),
                    linestyle='--',
                    linewidth=1.5,
                    alpha=reg.get('alpha', 0.8)
                )

                if reg.get('gamma'):
                    offset = -reg.get('offset', 45) * 1.6
                else:
                    offset = reg.get('offset', 5)

                plt.text(
                    reg['x'] + offset,
                    max(y) * reg.get('y_pos', 0.9),
                    reg.get('label', ''),
                    rotation=0,
                    color=reg.get('color_texto', 'black'),
                    fontsize=15,
                    va='bottom',
                    ha='left',
                    alpha=reg.get('alpha_text', 0.95),   # 💡 same visibility as region text
                    fontweight='bold'                    # 💡 same thickness as region text
                )

    # Estética general
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    if titulo:
        plt.title(titulo, fontsize=15)
    plt.legend(fontsize=14)
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
        carpeta = "./Experimento IV/Imagenes/BordeCompton"
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
        carpeta = "./Experimento IV/Imagenes/Gaussiana"
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


def ajustar_pico_gaussiano(x_data, y_data, x_err, y_err, p0, mostrarGrafica=True, nombre_archivo = 'None'):
    """Ajusta un pico gaussiano con ODR"""
    parametros, errores, output, gauss_ajustada = ajustar_gaussiana_odr(
        x_data, y_data, x_err, y_err, p0=p0, mostrar_grafica=mostrarGrafica, nombre_archivo = nombre_archivo
    )
    return parametros, errores, output, gauss_ajustada

def devolver_energia_cuentas(
    df,
    corte1=(13, 60),
    corteBa137=(13, 60),
    p0_1=[0, 30, 7, 4, 0],
    mostrarGrafica1=True,
    corte2=(550, 820),
    corteGamma=(540, 810),
    p0_2=[0, 662, 7, 4, 0],
    p0_Ba137 = [0, 33, 7, 4, 0],
    p0_Gamma = [0, 622, 7, 4, 0],
    mostrarGrafica2=True,
    mostrarGraficaFinal=True,
    corteRetro = (70, 120),
    corteCompton = (70, 120),
    p0_Compton = [0, 320, 8, 2],
    mostrarGraficaRetro = True, 
    mostrarGraficaCompton = True, 
    nombre_archivoRetro = 'Retro',
    nombre_archivoCompton = 'Compton',
    ajustarPlomo=False,
    cortePlomo=(850, 1050),
    p0_Plomo=None,
    mostrarGraficaPlomo=True,
    nombre_archivoPlomo="Plomo"
):
   # --- Definimos arrays base ---
    x = df["Canal"].values
    y = df["Cuentas"].values
    x_err = np.full(len(x), 1/2, dtype=float)
    y_err = np.sqrt(y)

    # --- Ajuste del primer pico ---
    x1, y1, xerr1, yerr1 = cortar_datos(*corte1, x, y, x_err, y_err)
    parametros1, errores1, _, _ = ajustar_pico_gaussiano(x1, y1, xerr1, yerr1, p0_1, mostrarGrafica1)

    # --- Ajuste del segundo pico ---
    x2, y2, xerr2, yerr2 = cortar_datos(*corte2, x, y, x_err, y_err)
    parametros2, errores2, _, _ = ajustar_pico_gaussiano(x2, y2, xerr2, yerr2, p0_2, mostrarGrafica2)


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
    

    errorX = np.full(len(df["Canal"][:800]), 1/2, dtype=float)
    Cuentas = df["Cuentas"][:800]
    errCuentas = np.sqrt(df["Cuentas"][:800])
    E, errE = calibrar(df["Canal"][:800], errorX, m, b, sm, sb)
      # --- Ajuste gaussiano + fondo lineal ---
    E_Gamma, Cuentas_Gamma, errE_Gamma, errCuentas_Gamma = cortar_datos(
        *corteGamma, E, Cuentas, errE, errCuentas
    )
     # --- Ajuste gaussiano + fondo lineal ---
    parametros_Gamma, errores_Gamma, _, _ = ajustar_pico_gaussiano(
        E_Gamma, Cuentas_Gamma, errE_Gamma, errCuentas_Gamma, p0_Gamma, False, nombre_archivoRetro
    )
    E_Ba137, Cuentas_Ba137, errE_Ba137, errCuentas_Ba137 = cortar_datos(
        *corteBa137, E, Cuentas, errE, errCuentas
    )
     # --- Ajuste gaussiano + fondo lineal ---
    parametros_Ba137, errores_Ba137, _, _ = ajustar_pico_gaussiano(
        E_Ba137, Cuentas_Ba137, errE_Ba137, errCuentas_Ba137, p0_Ba137, False, nombre_archivoRetro
    )
    regiones_cs137 = [
        {'tipo': 'pico', 'x': 662, 'label': 'Fotopico [662.0(1) keV]', 'color': 'black', 'alpha': 0.8, 'gamma':True, 'offset': 110},
        {'tipo': 'region', 'xmin': 440, 'xmax': 550, 'label': 'Borde Compton',
        'color': 'lightgreen', 'alpha': 0.25, 'alpha_text': 0.95},  # 💡 text stays solid
        {'tipo': 'region', 'xmin': 140, 'xmax': 300, 'label': 'Pico de retrodispersión',
        'color': 'lightblue', 'alpha': 0.25, 'alpha_text': 0.95},  # 💡 text stays solid
        {'tipo': 'pico', 'x': 32, 'label': 'Fotopico [32.0(2) keV]', 'color': 'black', 'alpha': 0.8, 'gamma':False}
    ]
    
    # print(f"Errores en E: {errE}")
    if mostrarGraficaFinal:
        graficar_con_error(E, Cuentas, errE, errCuentas, 'Energía [keV]', 'Cuentas',  regiones=regiones_cs137)
    
    E_retro, Cuentas_retro, errE_retro, errCuentas_retro = cortar_datos(
        *corteRetro, E, Cuentas, errE, errCuentas
    )

    # --- Estimaciones iniciales ---
    A0 = np.max(Cuentas_retro) - np.min(Cuentas_retro)
    mu0 = E_retro[np.argmax(Cuentas_retro)]
    sigma0 = 10  # ancho estimado (keV)
    m_lin0 = -2  # pendiente inicial negativa (fondo)
    b_lin0 = np.min(Cuentas_retro)
    p0_retro = [A0, mu0, sigma0, m_lin0, b_lin0]

    # --- Ajuste gaussiano + fondo lineal ---
    parametros_retro, errores_retro, _, _ = ajustar_pico_gaussiano(
        E_retro, Cuentas_retro, errE_retro, errCuentas_retro, p0_retro, mostrarGraficaRetro, nombre_archivoRetro
    )

    E_Compton, Cuentas_Compton, errE_Compton, errCuentas_Compton = cortar_datos(
        *corteCompton, E, Cuentas, errE, errCuentas
    )
    # --- Ajuste gaussiano + fondo lineal ---
    parametros_Compton, errores_Compton, _, _ = ajustar_borde_compton(
        E_Compton, Cuentas_Compton, errE_Compton, errCuentas_Compton, p0_Compton, mostrarGraficaCompton, nombre_archivoCompton
    )
    resultados_plomo = None
    if ajustarPlomo:
        E_Plomo, Cuentas_Plomo, errE_Plomo, errCuentas_Plomo = cortar_datos(
            *cortePlomo, E, Cuentas, errE, errCuentas
        )

        # Si no se proporcionan parámetros iniciales, estimamos automáticamente
        if p0_Plomo is None:
            A0 = np.max(Cuentas_Plomo) - np.min(Cuentas_Plomo)
            mu0 = E_Plomo[np.argmax(Cuentas_Plomo)]
            sigma0 = 10
            m_lin0 = -1
            b_lin0 = np.min(Cuentas_Plomo)
            p0_Plomo = [A0, mu0, sigma0, m_lin0, b_lin0]

        parametros_Plomo, errores_Plomo, _, _ = ajustar_pico_gaussiano(
            E_Plomo,
            Cuentas_Plomo,
            errE_Plomo,
            errCuentas_Plomo,
            p0_Plomo,
            mostrarGraficaPlomo,
            nombre_archivoPlomo,
        )

        resultados_plomo = {
            "parametros": parametros_Plomo,
            "errores": errores_Plomo,
        }
    # Retornamos resultados
    return E, errE, errCuentas, {
        "pico1": {"parametros": parametros1, "errores": errores1},
        "pico2": {"parametros": parametros2, "errores": errores2},
        "ajuste_lineal": {"m": m, "sm": sm, "b": b, "sb": sb},
    }