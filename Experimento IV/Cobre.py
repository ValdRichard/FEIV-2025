import numpy as np
from analisis import espectro, ajustar_gaussiana_odr, fit_lineal, graficar, calibrar, graficar_con_error

# --- Importamos datos del Plomo ---
ruta = "./Experimento IV/Datos/"
df = espectro(ruta, 'Cs137-cu.Spe')
def cortar_espectro(izquierda, derecha, p0,  mostrarGrafica):
    
    x_data = df["Canal"][izquierda:derecha]
    x_err = np.full(len(x_data), 1/1024, dtype=float)
    y_data = df["Cuentas"][izquierda:derecha]
    y_err = np.sqrt(y_data)

    # Ajuste
    parametros, errores, output, gauss_ajustada = ajustar_gaussiana_odr(
        x_data, y_data, x_err, y_err, p0=p0, mostrar_grafica=mostrarGrafica
    )

    print("Parámetros ajustados:", parametros)
    print("Errores estándar:", errores)
    return parametros, errores
    
# --- Primer pico -- 
parametros1, errores1 = cortar_espectro(13, 60,
        # beta[0] = amplitud
        # beta[1] = media
        # beta[2] = sigma
        # beta[3] = pendiente
        # beta[4] = ordenada
        p0=[0,30,7,4,0], mostrarGrafica=False)

# --- Segundo pico ---
parametros2, errores2 = cortar_espectro(550, 820,
        # beta[0] = amplitud
        # beta[1] = media
        # beta[2] = sigma
        # beta[3] = pendiente
        # beta[4] = ordenada
        p0=[0,662,7,4,0], mostrarGrafica=False)


# --- Código que calibra --- 
Energia = [33, 662]
errEnergia = [0.01, 0.01] 
canal = [parametros1[1], parametros2[1]]
errCanal = [errores1[1], errores2[1]]

m, sm, b, sb = fit_lineal(canal, Energia, errCanal, errEnergia)
errC = np.full(len(df["Canal"]), 1/1024, dtype=float)
E, errE = calibrar(df["Canal"], errC, m, b, sm, sb)

graficar(E, df["Cuentas"], 'Energia', 'Cuentas')
graficar_con_error(E, df["Cuentas"], errE, errC, 'Energia', 'Cuentas')