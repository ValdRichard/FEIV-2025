import numpy as np
from analisis import devolver_energia_cuentas, espectro, graficar

# --- Importamos datos del Plomo ---
ruta = "./Experimento IV/Datos/"
df = espectro(ruta, 'Cs137-madera.Spe')
# graficar(df["Canal"], df["Cuentas"], "Canal", "Cuentas")
#       beta[0] = amplitud
#     beta[1] = media
#     beta[2] = sigma
#     beta[3] = pendiente
#     beta[4] = ordenada
E, errE, errC, resultados = devolver_energia_cuentas(
    df,
    corte1=(10, 69),
    p0_1=[0, 35, 6, 3, 0],
    mostrarGrafica1=False,
    corte2=(540, 810),
    p0_2=[0, 660, 8, 4, 0],
    mostrarGrafica2=False,
    mostrarGraficaFinal=False,
    mostrarGraficoRetro=True, 
    corteRetro=(160, 280),
    p0_retro=[1, 214, 30, 4, 0]
)

print(resultados)