"""Visualizaciones del modelo de nivel de glucosa.

Genera las gráficas de análisis de Machine Learning y las exporta a
``visualizations/glucosa``:

- distribución del nivel de glucosa,
- dispersión de cada variable independiente frente a la glucosa,
- glucosa media por nivel de actividad física,
- matriz de correlación,
- valores reales frente a predichos,
- distribución de residuos,
- impacto relativo de cada variable (β estandarizado).

Uso:
    python -m src.glucosa.visualizacion
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import visualizacion as viz
from src.common.modelado import ModeloEntrenado

from . import limpieza, modelo
from .especificacion import ESPECIFICACION

ARCHIVOS_DISPERSION = {
    "Edad": "dispersion_edad.png",
    "IMC": "dispersion_imc.png",
    "Actividad_Fisica": "dispersion_actividad.png",
}


def graficar_distribucion(df: pd.DataFrame) -> Path:
    """Histograma de la distribución de ``Nivel_Glucosa``."""
    plt = viz.obtener_pyplot()
    figura, ejes = viz.nueva_figura(
        titulo="Distribución del nivel de glucosa",
        etiqueta_x=f"{ESPECIFICACION.especificacion_objetivo.etiqueta} "
        f"({ESPECIFICACION.especificacion_objetivo.unidad})",
        etiqueta_y="Frecuencia",
    )
    ejes.hist(df[ESPECIFICACION.objetivo], bins=45, color="#1f77b4", alpha=0.85,
              edgecolor="white")
    ejes.axvline(
        df[ESPECIFICACION.objetivo].mean(),
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"Media = {df[ESPECIFICACION.objetivo].mean():.2f}",
    )
    ejes.legend()

    ruta = viz.exportar(figura, ESPECIFICACION.clave, "distribucion_glucosa.png")
    viz.cerrar(figura)
    return ruta


def graficar_dispersiones(df: pd.DataFrame) -> list[Path]:
    """Dispersión de cada variable independiente frente a la glucosa."""
    rutas: list[Path] = []
    for variable in ESPECIFICACION.variables:
        especificacion = ESPECIFICACION.especificacion_de(variable)
        rutas.append(
            viz.dispersion_frente_objetivo(
                df,
                variable,
                ESPECIFICACION.objetivo,
                ESPECIFICACION.clave,
                etiqueta_variable=especificacion.etiqueta,
                etiqueta_objetivo=ESPECIFICACION.especificacion_objetivo.etiqueta,
                nombre_archivo=ARCHIVOS_DISPERSION[variable],
            )
        )
    return rutas


def graficar_perfil_actividad(df: pd.DataFrame) -> Path:
    """Glucosa media por nivel de actividad física.

    Permite ver de forma directa la relación inversa entre actividad física y
    nivel de glucosa detectada en el modelo.
    """
    plt = viz.obtener_pyplot()
    agrupado = df.groupby("Actividad_Fisica")[ESPECIFICACION.objetivo].mean()

    figura, ejes = viz.nueva_figura(
        titulo="Nivel de glucosa medio según actividad física",
        etiqueta_x="Nivel de actividad física (0 = sedentario, 9 = muy activo)",
        etiqueta_y=f"{ESPECIFICACION.especificacion_objetivo.etiqueta} medio "
        f"({ESPECIFICACION.especificacion_objetivo.unidad})",
    )
    ejes.bar(agrupado.index, agrupado.values, color="#2ca02c", alpha=0.85)
    ejes.plot(agrupado.index, agrupado.values, color="#d62728", marker="o",
              linewidth=2, label="Tendencia")
    ejes.set_xticks(list(agrupado.index))
    ejes.legend()

    ruta = viz.exportar(figura, ESPECIFICACION.clave, "glucosa_por_actividad.png")
    viz.cerrar(figura)
    return ruta


def graficar_correlacion(df: pd.DataFrame) -> Path:
    """Matriz de correlación entre las variables del modelo."""
    return viz.mapa_calor_correlacion(
        df,
        ESPECIFICACION.columnas,
        ESPECIFICACION.clave,
        titulo="Correlación entre variables (Glucosa)",
    )


def graficar_real_vs_predicho(entrenado: ModeloEntrenado) -> Path:
    """Dispersión de valores reales frente a predichos."""
    particion = entrenado.particion
    return viz.real_frente_predicho(
        particion["y_prueba"],
        entrenado.predecir(particion["X_prueba"]),
        ESPECIFICACION.clave,
        etiqueta_objetivo=ESPECIFICACION.especificacion_objetivo.etiqueta,
    )


def graficar_residuos(entrenado: ModeloEntrenado) -> Path:
    """Distribución y dispersión de los residuos."""
    particion = entrenado.particion
    return viz.grafica_residuos(
        particion["y_prueba"],
        entrenado.predecir(particion["X_prueba"]),
        ESPECIFICACION.clave,
        etiqueta_objetivo=ESPECIFICACION.especificacion_objetivo.etiqueta,
    )


def graficar_coeficientes(entrenado: ModeloEntrenado) -> Path:
    """Impacto relativo de cada variable (β estandarizado)."""
    return viz.barras_coeficientes(
        entrenado.coeficientes,
        ESPECIFICACION.clave,
        etiqueta_objetivo=ESPECIFICACION.especificacion_objetivo.etiqueta,
        titulo="Impacto de cada variable sobre el nivel de glucosa",
    )


def generar_todas(
    entrenado: ModeloEntrenado, df: pd.DataFrame | None = None
) -> list[Path]:
    """Genera y exporta todas las gráficas del modelo.

    Args:
        entrenado: Modelo entrenado.
        df: Dataset procesado; si se omite se carga desde ``data/processed``.

    Returns:
        Lista de rutas de las imágenes exportadas.
    """
    dataset = limpieza.cargar_procesado() if df is None else df

    rutas: list[Path] = [graficar_distribucion(dataset)]
    rutas.extend(graficar_dispersiones(dataset))
    rutas.append(graficar_perfil_actividad(dataset))
    rutas.append(graficar_correlacion(dataset))
    rutas.append(graficar_real_vs_predicho(entrenado))
    rutas.append(graficar_residuos(entrenado))
    rutas.append(graficar_coeficientes(entrenado))
    return rutas


def main() -> None:
    """Punto de entrada: ``python -m src.glucosa.visualizacion``."""
    entrenado = modelo.entrenar(persistir=False)
    for ruta in generar_todas(entrenado):
        print(ruta)


if __name__ == "__main__":
    main()
