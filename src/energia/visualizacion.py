"""Visualizaciones del modelo de consumo de energía.

Genera las gráficas de análisis de Machine Learning y las exporta a
``visualizations/energia``:

- perfil medio de consumo por hora,
- consumo medio por día de la semana,
- dispersión de cada variable independiente frente al consumo,
- comparación de la codificación de ``Hora`` (lineal frente a cíclica),
- matriz de correlación,
- valores reales frente a predichos,
- distribución de residuos,
- impacto relativo de cada variable (β estandarizado).

Uso:
    python -m src.energia.visualizacion
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import visualizacion as viz
from src.common.modelado import ModeloEntrenado

from . import limpieza, modelo
from .especificacion import ESPECIFICACION

ARCHIVOS_DISPERSION = {
    "Temperatura": "dispersion_temperatura.png",
    "Hora": "dispersion_hora.png",
    "Dia_Semana": "dispersion_dia_semana.png",
}

ETIQUETAS_DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def graficar_consumo_por_hora(df: pd.DataFrame) -> Path:
    """Perfil medio de ``Consumo_Energia`` por ``Hora``.

    La forma de esta curva es la evidencia que justifica la codificación lineal
    de la hora.
    """
    plt = viz.obtener_pyplot()
    agrupado = df.groupby("Hora")[ESPECIFICACION.objetivo].mean()

    figura, ejes = viz.nueva_figura(
        titulo="Consumo medio de energía por hora del día",
        etiqueta_x="Hora",
        etiqueta_y=f"{ESPECIFICACION.especificacion_objetivo.etiqueta} medio "
        f"({ESPECIFICACION.especificacion_objetivo.unidad})",
        tamano=(10, 4.8),
    )
    ejes.plot(agrupado.index, agrupado.values, color="#1f77b4", marker="o",
              linewidth=2, markersize=5)
    ejes.set_xticks(range(1, 25))

    ruta = viz.exportar(figura, ESPECIFICACION.clave, "consumo_por_hora.png")
    viz.cerrar(figura)
    return ruta


def graficar_consumo_por_dia(df: pd.DataFrame) -> Path:
    """Consumo medio por ``Dia_Semana``."""
    plt = viz.obtener_pyplot()
    agrupado = df.groupby("Dia_Semana")[ESPECIFICACION.objetivo].mean()

    figura, ejes = viz.nueva_figura(
        titulo="Consumo medio de energía por día de la semana",
        etiqueta_x="Día de la semana",
        etiqueta_y=f"{ESPECIFICACION.especificacion_objetivo.etiqueta} medio "
        f"({ESPECIFICACION.especificacion_objetivo.unidad})",
    )
    ejes.bar(agrupado.index, agrupado.values, color="#ff7f0e", alpha=0.85)
    ejes.set_xticks(list(agrupado.index))
    ejes.set_xticklabels(ETIQUETAS_DIAS)

    ruta = viz.exportar(figura, ESPECIFICACION.clave, "consumo_por_dia_semana.png")
    viz.cerrar(figura)
    return ruta


def graficar_dispersiones(df: pd.DataFrame) -> list[Path]:
    """Dispersión de cada variable independiente frente al consumo."""
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


def graficar_comparacion_codificacion(df: pd.DataFrame) -> Path:
    """Compara el R² de la codificación lineal y la cíclica de ``Hora``.

    Gráfica de evidencia: muestra por qué se eligió la escala lineal.
    """
    plt = viz.obtener_pyplot()
    puntuaciones = limpieza.comparar_codificaciones_hora(df)

    figura, ejes = viz.nueva_figura(
        titulo="Codificación de 'Hora': lineal vs. cíclica (seno/coseno)",
        etiqueta_x="Codificación",
        etiqueta_y="R² sobre el conjunto de prueba",
        tamano=(7, 5),
    )
    etiquetas = ["Lineal\n(usada)", "Cíclica\n(descartada)"]
    valores = [puntuaciones["hora_lineal"], puntuaciones["hora_ciclica"]]
    barras = ejes.bar(etiquetas, valores, color=["#2ca02c", "#d62728"], alpha=0.85)
    ejes.set_ylim(0, 1)

    for barra, valor in zip(barras, valores):
        ejes.text(barra.get_x() + barra.get_width() / 2, valor + 0.02, f"{valor:.4f}",
                  ha="center", fontsize=11, fontweight="bold")

    ruta = viz.exportar(
        figura, ESPECIFICACION.clave, "comparacion_codificacion_hora.png"
    )
    viz.cerrar(figura)
    return ruta


def graficar_correlacion(df: pd.DataFrame) -> Path:
    """Matriz de correlación entre las variables del modelo."""
    return viz.mapa_calor_correlacion(
        df,
        ESPECIFICACION.columnas,
        ESPECIFICACION.clave,
        titulo="Correlación entre variables (Energía)",
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
        titulo="Impacto de cada variable sobre el consumo de energía",
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

    rutas: list[Path] = [
        graficar_consumo_por_hora(dataset),
        graficar_consumo_por_dia(dataset),
    ]
    rutas.extend(graficar_dispersiones(dataset))
    rutas.append(graficar_comparacion_codificacion(dataset))
    rutas.append(graficar_correlacion(dataset))
    rutas.append(graficar_real_vs_predicho(entrenado))
    rutas.append(graficar_residuos(entrenado))
    rutas.append(graficar_coeficientes(entrenado))
    return rutas


def main() -> None:
    """Punto de entrada: ``python -m src.energia.visualizacion``."""
    entrenado = modelo.entrenar(persistir=False)
    for ruta in generar_todas(entrenado):
        print(ruta)


if __name__ == "__main__":
    main()
