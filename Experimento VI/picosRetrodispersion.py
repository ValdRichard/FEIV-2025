import numpy as np
import matplotlib.pyplot as plt
from analisis import devolver_energia_cuentas, espectro
import os
def graficar_retrodispersion_comparativa(configuraciones, titulo="Comparación de Picos de Retrodispersión"):
    """
    Grafica todos los picos de retrodispersión juntos para comparación
    
    Parameters:
    -----------
    configuraciones : list of dict
        Cada dict debe contener:
        - 'archivo': nombre del archivo .Spe
        - 'material': nombre del material (para la leyenda)
        - 'color': color para la gráfica
        - 'corteRetro': tupla con límites del pico de retrodispersión
        - otros parámetros necesarios para devolver_energia_cuentas
    titulo : str
        Título de la gráfica
    """
    
    plt.figure(figsize=(12, 8))
    ruta = "./Experimento IV/Datos/"
    
    for config in configuraciones:
        # Cargar y procesar datos
        df = espectro(ruta, config['archivo'])
        
        # Obtener energía y cuentas (sin mostrar gráficas individuales)
        E, errE, errC, resultados = devolver_energia_cuentas(
            df,
            corte1=config.get('corte1', (10, 69)),
            p0_1=config.get('p0_1', [0, 35, 6, 3, 0]),
            mostrarGrafica1=False,
            corte2=config.get('corte2', (540, 810)),
            p0_2=config.get('p0_2', [0, 660, 8, 4, 0]),
            mostrarGrafica2=False,
            mostrarGraficaFinal=False,
            mostrarGraficaRetro=False,  # Importante: no mostrar gráficas individuales
            mostrarGraficaCompton=False,
            corteRetro=config['corteRetro'],
            p0_Compton=config.get('p0_Compton', [0, 474, 8, 2]),
            corteCompton=config.get('corteCompton', (400, 550)),
            nombre_archivoRetro=config.get('material', 'temp'),
            nombre_archivoCompton=config.get('material', 'temp') + '_Compton'
        )
        
        # Extraer región del pico de retrodispersión
        inicio, fin = config['corteRetro']
        mascara = (E >= inicio) & (E <= fin)
        
        E_retro = E[mascara]
        cuentas_retro = df["Cuentas"][:len(E)][mascara]  # Asegurar misma longitud
        errE_retro = errE[mascara]
        errC_retro = errC[mascara]
        
        # Graficar con barras de error
        plt.errorbar(
            E_retro, cuentas_retro,
            xerr=errE_retro, yerr=errC_retro,
            fmt='o', markersize=3, capsize=2,
            label=config['material'],
            color=config.get('color', None),
            alpha=0.7
        )
    
    plt.xlabel('Energía [keV]', fontsize=12)
    plt.ylabel('Cuentas', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Guardar la gráfica comparativa
    carpeta = "./Experimento IV/Imagenes/Comparativas/"
    os.makedirs(carpeta, exist_ok=True)
    plt.savefig(f"{carpeta}/Retrodispersion_Comparativa.png", dpi=300)
    
    plt.show()

# Ejemplo de uso con tus configuraciones:
configuraciones = [
    {
        'archivo': 'Cs137-cu.Spe',
        'material': 'Cobre',
        'color': 'peru',
        'corteRetro': (140, 300),
        'corte1': (10, 69),
        'p0_1': [0, 35, 6, 3, 0],
        'corte2': (540, 810),
        'p0_2': [0, 660, 8, 4, 0],
        'p0_Compton': [0, 474, 8, 2],
        'corteCompton': (400, 550)
    },
    {
        'archivo': 'Cs137-Pb.Spe',  # Ajusta los nombres de archivo
        'material': 'Plomo',
        'color': 'grey',
        'corteRetro': (140, 300),  # Ajusta según tus datos
        'corte1': (10, 69),
        'p0_1': [0, 35, 6, 3, 0],
        'corte2': (540, 810),
        'p0_2': [0, 660, 8, 4, 0],
        'p0_Compton': [0, 474, 8, 2],
        'corteCompton': (400, 550)
    },
    {
        'archivo': 'Cs137-madera.Spe',  # Ajusta los nombres de archivo
        'material': 'Madera',
        'color': 'brown',
        'corteRetro': (140, 300),  # Ajusta según tus datos
        'corte1': (10, 69),
        'p0_1': [0, 35, 6, 3, 0],
        'corte2': (540, 810),
        'p0_2': [0, 660, 8, 4, 0],
        'p0_Compton': [0, 474, 8, 2],
        'corteCompton': (400, 550)
    },
    {
        'archivo': 'Cs137-2.Spe',  # Ajusta los nombres de archivo
        'material': 'Laboratorio',
        'color': 'blue',
        'corteRetro': (140, 300),  # Ajusta según tus datos
        'corte1': (10, 69),
        'p0_1': [0, 35, 6, 3, 0],
        'corte2': (540, 810),
        'p0_2': [0, 660, 8, 4, 0],
        'p0_Compton': [0, 474, 8, 2],
        'corteCompton': (400, 550)
    }
]

# Ejecutar la comparación
graficar_retrodispersion_comparativa(configuraciones)