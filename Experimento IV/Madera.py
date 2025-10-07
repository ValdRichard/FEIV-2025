import numpy as np
from analisis import devolver_energia_cuentas, espectro, ajustar_gaussiana_odr, fit_lineal, graficar, calibrar, graficar_con_error

# --- Importamos datos del Plomo ---
ruta = "./Experimento IV/Datos/"
df = espectro(ruta, 'Cs137-madera.Spe')

E, errE, errC, resultados = devolver_energia_cuentas(
    df,
    corte1=(20, 70),
    p0_1=[0, 35, 6, 3, 0],
    mostrarGrafica1=True,
    corte2=(540, 810),
    p0_2=[0, 660, 8, 4, 0],
    mostrarGrafica2=True,
    mostrarGraficaFinal=True,
)

print(resultados)