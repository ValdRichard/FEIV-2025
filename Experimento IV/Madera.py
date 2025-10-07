import numpy as np
from analisis import devolver_energia_cuentas, espectro

# --- Importamos datos del Plomo ---
ruta = "./Experimento IV/Datos/"
df = espectro(ruta, 'Cs137-madera.Spe')

E, errE, errC, resultados = devolver_energia_cuentas(
    df,
    corte1=(4, 70),
    p0_1=[0, 35, 6, 3, 0],
    mostrarGrafica1=True,
    corte2=(540, 810),
    p0_2=[0, 660, 8, 4, 0],
    mostrarGrafica2=True,
    mostrarGraficaFinal=True,
    corteRetro = (70, 120),
    p0_retro = [0, 320, 8, 4, 0],
    corteCompton = (70, 120),
    p0_compton = [0, 320, 8, 4, 0]
)