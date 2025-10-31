import numpy as np
from analisis import devolver_energia_cuentas, espectro

# --- Importamos datos del Cesio ---
ruta = "./Experimento V/Datos/"
df = espectro(ruta, 'Cs137-18-9-1800s.Spe')

E, errE, errC, resultados = devolver_energia_cuentas(
    df,
    corte1=(3, 36),
    p0_1=[0, 15, 6, 3, 0],
    mostrarGrafica1=True,
    corte2=(350, 750),
    p0_2=[0, 510, 8, 4, 0],
    mostrarGrafica2=True,
    mostrarGraficaFinal=True,
    mostrarGraficaRetro=False,
    mostrarGraficaCompton=False, 
    corteRetro=(140, 300),
    p0_Compton = [0, 474, 8, 2],
    corteCompton=(400, 550),
    nombre_archivoRetro = 'RetroCesio',
    nombre_archivoCompton = 'ComptonCesio'
)
