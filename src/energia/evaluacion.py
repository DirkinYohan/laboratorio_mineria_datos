"""Evaluación del modelo de consumo de energía.

Carga el modelo exportado, hace predicciones sobre el dataset procesado y
calcula las métricas exigidas por el laboratorio: **RMSE** y **R²** (se añaden
MAE, MSE y R² ajustado como apoyo a la interpretación). También obtiene los
coeficientes y el intercepto, y guarda el análisis en ``results/energia``.

Uso:
    python -m src.energia.evaluacion
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common import io
from src.common.metricas import (
    calcular_metricas,
    interpretar_desempeno,
    r2_ajustado,
    resumen_residuos,
    tabla_predicciones,
)

from src.common.modelado import dividir_datos
from . import limpieza
from .especificacion import ESPECIFICACION


def cargar_modelo_exportado() -> Any:
    """Carga el artefacto ``.joblib`` del modelo.

    Raises:
        FileNotFoundError: Si el modelo aún no ha sido entrenado.
    """
    return io.cargar_modelo(ESPECIFICACION)


def preparar_datos() -> pd.DataFrame:
    """Carga el dataset procesado y conserva las columnas del modelo."""
    df = limpieza.cargar_procesado()
    return df.loc[:, list(ESPECIFICACION.columnas)]


def predecir(estimador: Any, X: pd.DataFrame) -> list[float]:
    """Genera predicciones para la matriz de variables indicada.

    Args:
        estimador: Modelo cargado con ``joblib``.
        X: Variables independientes en el orden declarado.

    Returns:
        Lista de predicciones como números flotantes.

    Raises:
        ValueError: Si faltan variables o el número de columnas no coincide.
    """
    faltantes = [c for c in ESPECIFICACION.variables if c not in X.columns]
    if faltantes:
        raise ValueError(f"Faltan variables para predecir: {faltantes}")

    X = X.loc[:, list(ESPECIFICACION.variables)]
    esperadas = getattr(estimador, "n_features_in_", len(ESPECIFICACION.variables))
    if X.shape[1] != esperadas:
        raise ValueError(
            f"El modelo espera {esperadas} variables, pero se recibieron {X.shape[1]}."
        )

    return [float(valor) for valor in estimador.predict(X)]


def _reales_y_predichos(estimador: Any) -> tuple[pd.Series, list[float]]:
    """Devuelve (valores reales, valores predichos) sobre el dataset procesado."""
    df = preparar_datos()
    X = df.loc[:, list(ESPECIFICACION.variables)]
    return df[ESPECIFICACION.objetivo], predecir(estimador, X)


def evaluar(estimador: Any | None = None) -> dict[str, Any]:
    """Calcula las métricas del modelo exportado.

    Se reportan **dos conjuntos de métricas**, porque miden cosas distintas y
    conviene no confundirlas:

    - ``metricas_prueba``: sobre el 20 % reservado (partición 80/20 con semilla
      42). Es la medida honesta de la capacidad de generalización.
    - ``metricas_completo``: sobre el dataset procesado completo.

    Incluye además la comparación contra una **línea base ingenua**.

    Args:
        estimador: Modelo ya cargado; si se omite se carga desde disco.

    Returns:
        Diccionario con ``metricas``, ``metricas_prueba``, ``metricas_completo``,
        ``linea_base`` y ``mejora_rmse_vs_linea_base``.
    """
    model = estimador if estimador is not None else cargar_modelo_exportado()
    df = preparar_datos()
    X = df.loc[:, list(ESPECIFICACION.variables)]
    y = df[ESPECIFICACION.objetivo]

    y_predicho = predecir(model, X)
    completo = calcular_metricas(y, y_predicho)
    completo["r2_ajustado"] = r2_ajustado(
        completo["r2"], int(completo["n_muestras"]), len(ESPECIFICACION.variables)
    )

    X_ent, X_prueba, y_ent, y_prueba = dividir_datos(X, y)
    pred_prueba = predecir(model, X_prueba)
    prueba = calcular_metricas(y_prueba, pred_prueba)
    prueba["r2_ajustado"] = r2_ajustado(
        prueba["r2"], int(prueba["n_muestras"]), len(ESPECIFICACION.variables)
    )

    media = float(pd.Series(y_ent).mean())
    base = calcular_metricas(y_prueba, [media] * len(y_prueba))
    base["r2"] = 0.0

    mejora = 1.0 - (prueba["rmse"] / base["rmse"]) if base["rmse"] else 0.0

    return {
        "metricas": prueba,
        "metricas_prueba": prueba,
        "metricas_completo": completo,
        "linea_base": base,
        "mejora_rmse_vs_linea_base": float(mejora),
    }


def obtener_coeficientes(estimador: Any | None = None) -> dict[str, Any]:
    """Obtiene el intercepto y los coeficientes del modelo exportado.

    Returns:
        ``{"intercepto": float, "coeficientes": {variable: valor}}``.
    """
    model = estimador if estimador is not None else cargar_modelo_exportado()

    return {
        "intercepto": float(getattr(model, "intercept_", 0.0)),
        "coeficientes": {
            variable: float(coeficiente)
            for variable, coeficiente in zip(
                ESPECIFICACION.variables, getattr(model, "coef_", [])
            )
        },
    }


def analizar(estimador: Any | None = None) -> dict[str, Any]:
    """Reúne métricas, coeficientes, interpretación y análisis de residuos."""
    model = estimador if estimador is not None else cargar_modelo_exportado()
    y, y_predicho = _reales_y_predichos(model)
    evaluacion = evaluar(model)
    ajuste = obtener_coeficientes(model)

    return {
        "modelo": ESPECIFICACION.clave,
        "nombre": ESPECIFICACION.nombre,
        "variable_objetivo": ESPECIFICACION.objetivo,
        "unidad_objetivo": ESPECIFICACION.especificacion_objetivo.unidad,
        "variables": list(ESPECIFICACION.variables),
        "intercepto": ajuste["intercepto"],
        "coeficientes": ajuste["coeficientes"],
        "evaluacion": evaluacion,
        "interpretacion_desempeno": interpretar_desempeno(
            evaluacion["metricas"],
            ESPECIFICACION.especificacion_objetivo.etiqueta,
            ESPECIFICACION.especificacion_objetivo.unidad,
        ),
        "residuos": resumen_residuos(y, y_predicho),
    }


def guardar_reporte(estimador: Any | None = None) -> Path:
    """Persiste métricas, análisis, predicciones y residuos.

    Artefactos generados en ``results/energia``:
        - ``metricas.json``      RMSE, R², MAE, MSE y R² ajustado
        - ``analisis.json``      interpretación, coeficientes e intercepto
        - ``predicciones.csv``   reales frente a predichos con residuos
        - ``residuos.json``      estadísticos de los residuos
    """
    model = estimador if estimador is not None else cargar_modelo_exportado()
    y, y_predicho = _reales_y_predichos(model)

    io.guardar_json(evaluar(model), ESPECIFICACION.clave, "metricas.json")
    io.guardar_json(analizar(model), ESPECIFICACION.clave, "analisis.json")
    io.guardar_tabla(
        tabla_predicciones(y, y_predicho), ESPECIFICACION.clave, "predicciones.csv"
    )
    io.guardar_json(
        resumen_residuos(y, y_predicho), ESPECIFICACION.clave, "residuos.json"
    )
    return io.directorio_resultados(ESPECIFICACION.clave)


def main() -> None:
    """Punto de entrada: ``python -m src.energia.evaluacion``."""
    reporte = analizar()
    guardar_reporte()

    metricas = reporte["evaluacion"]["metricas"]
    print(f"[{ESPECIFICACION.clave}] {reporte['interpretacion_desempeno']}")
    print(f"[{ESPECIFICACION.clave}] RMSE       = {metricas['rmse']:.4f}")
    print(f"[{ESPECIFICACION.clave}] R2         = {metricas['r2']:.6f}")
    print(f"[{ESPECIFICACION.clave}] Intercepto = {reporte['intercepto']:.4f}")


if __name__ == "__main__":
    main()
