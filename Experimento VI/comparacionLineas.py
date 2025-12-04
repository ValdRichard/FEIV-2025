import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.odr import ODR, Model, RealData

# ============================================
# MODELO LINEAL PARA ODR
# ============================================
def lineal(B, x):
    a, b = B
    return a*x + b

model = Model(lineal)
MIN_ERROR = 1e-6

# ============================================
# FUNCIÓN GENERAL PARA AJUSTAR UN ARCHIVO
# ============================================
def procesar_archivo(archivo, prefijo):
    print(f"\n===================================")
    print(f"   Procesando {prefijo} desde {archivo}")
    print(f"===================================")

    df = pd.read_csv(archivo)

    # Detectar columnas tipo Kb1, Lb2, etc.
    k_cols   = sorted([c for c in df.columns if c.startswith(prefijo) and c[len(prefijo):].isdigit()])
    err_cols = sorted([c for c in df.columns if c.startswith("err" + prefijo)])
    z_cols   = sorted([c for c in df.columns if c.startswith("Z" + prefijo) or c.startswith("Z" + prefijo.lower())])

    results_array = []

    # Emparejar por grupo (1,2,3...)
    grupos = sorted([int(c[len(prefijo):]) for c in k_cols])

    for g in grupos:

        Kcol   = f"{prefijo}{g}"
        errcol = f"err{prefijo}{g}"
        Zcol1  = f"Z{prefijo}{g}"
        Zcol2  = f"Z{prefijo.lower()}{g}"
        Zcol = Zcol1 if Zcol1 in df.columns else Zcol2

        if Kcol not in df.columns or errcol not in df.columns or Zcol not in df.columns:
            print(f"⚠ Grupo {g} incompleto → se omite {prefijo}{g}")
            continue

        Z  = df[Zcol].values
        K  = df[Kcol].values
        eK = df[errcol].values

        mask = (~np.isnan(Z)) & (~np.isnan(K)) & (~np.isnan(eK)) & (eK > 0)
        Zc, Kc, eKc = Z[mask], K[mask], eK[mask]

        if len(Zc) < 2:
            continue

        eKc[eKc < MIN_ERROR] = MIN_ERROR

        # --- Ajuste ODR ---
        data = RealData(Zc, Kc, sy=eKc)
        odr = ODR(data, model, beta0=[0.1, -0.2])
        out = odr.run()

        a, b = out.beta
        ea, eb = out.sd_beta

        results_array.append({
            "serie": Kcol,
            "a": a,
            "ea": ea,
            "b": b,
            "eb": eb
        })

        # --- FIGURA INDIVIDUAL ---
        plt.figure(figsize=(6,4))
        plt.errorbar(Zc, Kc, yerr=eKc, fmt='o', label=f"{Kcol} puntos")
        xfit = np.linspace(min(Zc), max(Zc), 200)
        yfit = a * xfit + b
        plt.plot(xfit, yfit, label=f"{Kcol} recta (a={a:.5f})")

        plt.xlabel("Z")
        plt.ylabel(prefijo)
        plt.title(f"Ajuste lineal de {Kcol}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    print("\nRESULTADOS PARA", prefijo)
    for item in results_array:
        print(item)

    return results_array


# === PROCESAR ARCHIVOS ===

resultados_Ka = procesar_archivo(
    r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - Ka.csv", "Ka"
)

# resultados_Kb = procesar_archivo(
#     r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - Kb.csv", "Kb"
# )
# resultados_La = procesar_archivo(
#     r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - La.csv", "La"
# )
# resultados_Lb = procesar_archivo(
#     r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - Lb.csv", "Lb"
# )
# resultados_Lg = procesar_archivo(
#     r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - Lg.csv", "Lg"
# )
def grafico_por_prefijo(archivo, prefijo, resultados):

    df = pd.read_csv(archivo)

    k_cols   = sorted([c for c in df.columns if c.startswith(prefijo) and c[len(prefijo):].isdigit()])
    err_cols = sorted([c for c in df.columns if c.startswith("err" + prefijo)])
    z_cols   = sorted([c for c in df.columns if c.startswith("Z" + prefijo) or c.startswith("Z" + prefijo.lower())])

    grupos = sorted([int(c[len(prefijo):]) for c in k_cols])

    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    # Crear una lista de colores
    colores = plt.cm.tab10(np.linspace(0, 1, len(grupos)))

    # ============================
    # RECORRER CADA GRUPO (1..N)
    # ============================
    for idx, g in enumerate(grupos):

        color = colores[idx]

        Kcol   = f"{prefijo}{g}"
        errcol = f"err{prefijo}{g}"
        Zcol1  = f"Z{prefijo}{g}"
        Zcol2  = f"Z{prefijo.lower()}{g}"
        Zcol = Zcol1 if Zcol1 in df.columns else Zcol2

        if Kcol not in df.columns or errcol not in df.columns or Zcol not in df.columns:
            print(f"⚠ Falta info en {prefijo}{g}. Se omite.")
            continue

        Z  = df[Zcol].values
        K  = df[Kcol].values
        eK = df[errcol].values

        mask = (~np.isnan(Z)) & (~np.isnan(K)) & (~np.isnan(eK)) & (eK > 0)
        Zc, Kc, eKc = Z[mask], K[mask], eK[mask]

        if len(Zc) < 2:
            continue

        # ========= PUNTOS =========
        ax.errorbar(
            Zc, Kc, yerr=eKc,
            fmt='o',
            markersize=6,
            color=color,
            markeredgecolor='black',   # borde negro
            markeredgewidth=1.2,        # borde más grueso
            label=f"{Kcol} puntos",
            alpha=0.85
        )

        # ========= BUSCAR AJUSTE =========
        r = next((rr for rr in resultados if rr["serie"] == Kcol), None)
        if r is None:
            continue

        a, b = r["a"], r["b"]

        # ========= RECTA =========
        xfit = np.linspace(min(Zc), max(Zc), 200)
        yfit = a * xfit + b

        ax.plot(
            xfit, yfit,
            linewidth=2.5,
            color=color,
            alpha=0.9,
            label=f"{Kcol} recta (a={a:.4f})"
        )

    plt.title(f"Todas las series del prefijo {prefijo}")
    plt.xlabel("Z")
    plt.ylabel(prefijo)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# === GRAFICOS FINALES ===

# grafico_por_prefijo(r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - Kb.csv", "Kb", resultados_Kb)
grafico_por_prefijo(r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - Ka.csv", "Ka", resultados_Ka)
# grafico_por_prefijo(r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - La.csv", "La", resultados_La)
# grafico_por_prefijo(r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - Lb.csv", "Lb", resultados_Lb)
# grafico_por_prefijo(r"E:\FisicaTP\FisicaExperimentalIV\FEIV-2025\Experimento VI\Datos\Presentacion tp6 - Lg.csv", "Lg", resultados_Lg)
