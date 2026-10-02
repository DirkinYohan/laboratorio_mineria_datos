"""Visualizaciones del modelo de precio del dólar.

Genera las gráficas de análisis de Machine Learning y las exporta a
``visualizations/dolar``:

- evolución del precio a lo largo del tiempo,
- dispersión de cada variable independiente frente al precio,
- matriz de correlación,
- valores reales frente a predichos,
- distribución de residuos,
- impacto relativo de cada variable (β estandarizado).

Uso:
    python -m src.dolar.visualizacion
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import visualizacion as viz
from src.common.modelado import ModeloEntrenado

from . import limpieza, modelo
from .especificacion import ESPECIFICACION

ARCHIVOS_DISPERSION = {
    "Dia": "dispersion_dia.png",
    "Inflacion": "dispersion_inflacion.png",
    "Tasa_interes": "dispersion_tasa_interes.png",
}


def graficar_serie(df: pd.DataFrame) -> Path:
    """Evolución del ``Precio_Dolar`` a lo largo de ``Dia``."""
    plt = viz.obtener_pyplot()
    figura, ejes = viz.nueva_figura(
        titulo="Evolución del precio del dólar",
        etiqueta_x="Día",
        etiqueta_y="Precio del dólar",
        tamano=(10, 4.8),
    )
    ejes.plot(df["Dia"], df["Precio_Dolar"], color="#1f77b4", linewidth=1.6)
    ejes.fill_between(
        df["Dia"],
        df["Precio_Dolar"],
        df["Precio_Dolar"].min(),
        color="#1f77b4",
        alpha=0.12,
    )

    ruta = viz.exportar(figura, ESPECIFICACION.clave, "serie_precio.png")
    viz.cerrar(figura)
    return ruta


def graficar_dispersiones(df: pd.DataFrame) -> list[Path]:
    """Dispersión de cada variable independiente frente al precio."""
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


def graficar_correlacion(df: pd.DataFrame) -> Path:
    """Matriz de correlación entre las variables del modelo."""
    return viz.mapa_calor_correlacion(
        df,
        ESPECIFICACION.columnas,
        ESPECIFICACION.clave,
        titulo="Correlación entre variables (Dólar)",
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
        titulo="Impacto de cada variable sobre el precio del dólar",
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

    rutas: list[Path] = [graficar_serie(dataset)]
    rutas.extend(graficar_dispersiones(dataset))
    rutas.append(graficar_correlacion(dataset))
    rutas.append(graficar_real_vs_predicho(entrenado))
    rutas.append(graficar_residuos(entrenado))
    rutas.append(graficar_coeficientes(entrenado))
    return rutas


def main() -> None:
    """Punto de entrada: ``python -m src.dolar.visualizacion``."""
    entrenado = modelo.entrenar(persistir=False)
    for ruta in generar_todas(entrenado):
        print(ruta)


if __name__ == "__main__":
    main()
