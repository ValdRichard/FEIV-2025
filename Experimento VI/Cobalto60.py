import numpy as np
from analisis import devolver_energia_cuentas, espectro, graficar


# --- Importamos datos del Cobalto ---
ruta = "./Experimento V/Datos/"
df = espectro(ruta, 'Co60-18-9.Spe')
# graficar(df["Canal"], df["Cuentas"], "Canal", "Cuentas")
#       beta[0] = amplitud
#     beta[1] = media
#     beta[2] = sigma
#     beta[3] = pendiente
#     beta[4] = ordenada
E, errE, errC, resultados = devolver_energia_cuentas(
    df,
    corte1=(50, 250),
    p0_1=[0, 130, 6, 3, 0],
    mostrarGrafica1=True,
    corte2=(600, 750),
    p0_2=[0, 680, 8, 4, 0],
    mostrarGrafica2=True,
    mostrarGraficaFinal=True,
    mostrarGraficaRetro=False,
    mostrarGraficaCompton=False, 
    corteRetro=(140, 300),
    p0_Compton = [0, 474, 8, 2],
    corteCompton=(400, 550),
    nombre_archivoRetro = 'RetroCobalto',
    nombre_archivoCompton = 'ComptonCobalto'
)