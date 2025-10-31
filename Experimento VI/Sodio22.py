from analisis import devolver_energia_cuentas, espectro, graficar

# --- Importamos datos del Sodio ---
ruta = "./Experimento V/Datos/"
df = espectro(ruta, 'Na22-18-9-1800s.Spe')
E, errE, errC, resultados = devolver_energia_cuentas(
    df,
    corte1=(4, 70),
    p0_1=[0, 35, 6, 3, 0],
    mostrarGrafica1=False,
    corte2=(540, 810),
    p0_2=[0, 660, 8, 4, 0],
    mostrarGrafica2=False,
    mostrarGraficaFinal=False,
    mostrarGraficaRetro=True,
    mostrarGraficaCompton=True, 
    corteRetro=(140, 300),
    p0_Compton = [0, 474, 8, 2],
    corteCompton=(400, 550),
    nombre_archivoRetro = 'RetroSodio',
    nombre_archivoCompton = 'ComptonSodio'
)