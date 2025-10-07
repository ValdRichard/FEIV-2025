import numpy as np
from analisis import espectro, ajustar_gaussiana_odr, fit_lineal, graficar, calibrar, graficar_con_error

# --- Importamos datos del Plomo ---
ruta = "./Experimento IV/Datos/"
df = espectro(ruta, 'Cs137-madera.Spe')

def devolver_energia_cuentas(
    df,
    corte1=(13, 60),
    p0_1=[0, 30, 7, 4, 0],
    mostrarGrafica1=True,
    corte2=(550, 820),
    p0_2=[0, 662, 7, 4, 0],
    mostrarGrafica2=True,
    mostrarGraficaFinal=True,
):
    def cortar_espectro(izquierda, derecha, p0, mostrarGrafica):
        x_data = df["Canal"][izquierda:derecha]
        x_err = np.full(len(x_data), 1/1024, dtype=float)
        y_data = df["Cuentas"][izquierda:derecha]
        y_err = np.sqrt(y_data)

        parametros, errores, output, gauss_ajustada = ajustar_gaussiana_odr(
            x_data, y_data, x_err, y_err, p0=p0, mostrar_grafica=mostrarGrafica
        )

        print("Parámetros ajustados:", parametros)
        print("Errores estándar:", errores)
        return parametros, errores

    # --- Ajuste de los dos picos ---
    parametros1, errores1 = cortar_espectro(*corte1, p0_1, mostrarGrafica1)
    parametros2, errores2 = cortar_espectro(*corte2, p0_2, mostrarGrafica2)

    # --- Calibración ---
    Energia = [33, 662]
    errEnergia = [1, 1]
    canal = [parametros1[1], parametros2[1]]
    errCanal = [errores1[1], errores2[1]]

    m, sm, b, sb = fit_lineal(canal, Energia, errCanal, errEnergia)

    errC = np.full(len(df["Canal"]), 1/1024, dtype=float)
    E, errE = calibrar(df["Canal"], errC, m, b, sm, sb)

    if mostrarGraficaFinal:
        graficar_con_error(E, df["Cuentas"], errE, errC, 'Energía (keV)', 'Cuentas')

    # Retornamos resultados
    return E, errE, errC, {
        "pico1": {"parametros": parametros1, "errores": errores1},
        "pico2": {"parametros": parametros2, "errores": errores2},
        "ajuste_lineal": {"m": m, "sm": sm, "b": b, "sb": sb},
    }

E, errE, errC, resultados = devolver_energia_cuentas(
    df,
    corte1=(20, 70),
    p0_1=[0, 35, 6, 3, 0],
    mostrarGrafica1=True,
    corte2=(540, 810),
    p0_2=[0, 660, 8, 4, 0],
    mostrarGrafica2=False,
    mostrarGraficaFinal=True,
)

print(resultados)