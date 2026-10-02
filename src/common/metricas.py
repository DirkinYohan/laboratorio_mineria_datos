"""Métricas de evaluación de los modelos de regresión.

Calcula las métricas estándar, las compara contra una línea base ingenua y
construye el texto interpretativo que alimenta el informe.

Métricas exigidas por el laboratorio:
    - Dólar: MSE y R²
    - Glucosa: MSE y R²
    - Energía: RMSE y R²

Además se calculan MAE y R² ajustado porque ayudan a interpretar el desempeño,
y el RMSE (o MSE) de cada modelo está en las mismas unidades que su variable
objetivo, lo que facilita la lectura.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def calcular_metricas(y_real: Any, y_predicho: Any) -> dict[str, float]:
    """Calcula las métricas de evaluación de un modelo de regresión.

    Returns:
        ``{"r2", "mae", "mse", "rmse", "n_muestras"}``.
    """
    real = np.asarray(y_real, dtype=float).ravel()
    predicho = np.asarray(y_predicho, dtype=float).ravel()
    mse = float(mean_squared_error(real, predicho))

    return {
        "r2": float(r2_score(real, predicho)),
        "mae": float(mean_absolute_error(real, predicho)),
        "mse": mse,
        "rmse": float(root_mean_squared_error(real, predicho)),
        "n_muestras": float(real.size),
    }


def r2_ajustado(r2: float, n_muestras: int, n_variables: int) -> float:
    """Calcula el R² ajustado, que penaliza el número de variables.

    Raises:
        ValueError: Si no hay grados de libertad suficientes.
    """
    if n_muestras - n_variables - 1 <= 0:
        raise ValueError("No hay grados de libertad suficientes para el R² ajustado.")
    return 1.0 - (1.0 - r2) * (n_muestras - 1) / (n_muestras - n_variables - 1)


def metricas_linea_base(y_entrenamiento: Any, y_prueba: Any) -> dict[str, float]:
    """Métricas de una línea base ingenua: predecir la media del entrenamiento.

    Sirve para responder a la pregunta «¿el modelo aporta algo?»: un modelo útil
    debe superar claramente esta referencia.
    """
    media = float(np.asarray(y_entrenamiento, dtype=float).ravel().mean())
    prueba = np.asarray(y_prueba, dtype=float).ravel()

    metricas = calcular_metricas(prueba, np.full_like(prueba, media))
    metricas["r2"] = 0.0  # Predecir la media no explica varianza por definición.
    return metricas


def evaluar_modelo(modelo: Any, n_variables: int | None = None) -> dict[str, Any]:
    """Evalúa un ``ModeloEntrenado`` y enriquece el resultado con contexto.

    Añade el R² ajustado, las métricas de la línea base y la ganancia del modelo
    frente a esa línea base.
    """
    particion = modelo.particion
    y_prueba = particion["y_prueba"]
    y_entrenamiento = particion["y_entrenamiento"]
    y_predicho = modelo.predecir(particion["X_prueba"])

    metricas = calcular_metricas(y_prueba, y_predicho)
    n_variables = n_variables if n_variables is not None else len(modelo.variables)

    metricas["r2_ajustado"] = r2_ajustado(
        metricas["r2"], int(metricas["n_muestras"]), n_variables
    )

    base = metricas_linea_base(y_entrenamiento, y_prueba)
    mejora = 1.0 - (metricas["rmse"] / base["rmse"]) if base["rmse"] else 0.0

    return {
        "metricas": metricas,
        "linea_base": base,
        "mejora_rmse_vs_linea_base": float(mejora),
    }


def tabla_predicciones(y_real: Any, y_predicho: Any, n: int = 20) -> pd.DataFrame:
    """Tabla de valores reales frente a predichos, con su residuo."""
    real = np.asarray(y_real, dtype=float).ravel()
    predicho = np.asarray(y_predicho, dtype=float).ravel()

    return pd.DataFrame(
        {
            "real": real,
            "predicho": predicho,
            "residuo": real - predicho,
            "error_absoluto": np.abs(real - predicho),
        }
    ).head(n)


def resumen_residuos(y_real: Any, y_predicho: Any) -> dict[str, float]:
    """Resumen estadístico de los residuos para validar los supuestos del modelo.

    Incluye la media (debería ser ~0), la desviación estándar, los extremos y la
    forma de la distribución (asimetría y curtosis).
    """
    residuos = np.asarray(y_real, dtype=float).ravel() - np.asarray(
        y_predicho, dtype=float
    ).ravel()
    serie = pd.Series(residuos)

    return {
        "residuo_medio": float(serie.mean()),
        "residuo_std": float(serie.std()),
        "residuo_min": float(serie.min()),
        "residuo_max": float(serie.max()),
        "asimetria": float(serie.skew()),
        "curtosis": float(serie.kurtosis()),
    }


def interpretar_desempeno(
    metricas: Mapping[str, float], etiqueta_objetivo: str, unidad: str
) -> str:
    """Redacta una interpretación del desempeño a partir de las métricas reales.

    El texto se deriva de los valores calculados: no hay frases fijas.
    """
    r2 = metricas.get("r2", 0.0)
    rmse = metricas.get("rmse", 0.0)
    mae = metricas.get("mae", 0.0)

    if r2 >= 0.95:
        calidad, detalle = "excelente", (
            "el modelo reproduce casi por completo la variabilidad observada"
        )
    elif r2 >= 0.85:
        calidad, detalle = "muy bueno", (
            "el modelo explica la mayor parte de la variabilidad del fenómeno"
        )
    elif r2 >= 0.70:
        calidad, detalle = "bueno", (
            "el modelo explica una proporción sustancial de la variabilidad"
        )
    elif r2 >= 0.50:
        calidad, detalle = "moderado", (
            "el modelo captura la tendencia principal, pero deja variabilidad "
            "sin explicar"
        )
    else:
        calidad, detalle = "limitado", (
            "el modelo explica solo una parte reducida de la variabilidad"
        )

    return (
        f"El ajuste es {calidad}: R² = {r2:.4f}, es decir, {detalle} "
        f"({r2 * 100:.2f} % de la varianza de {etiqueta_objetivo}). En promedio, "
        f"el error de predicción es de {mae:.4f} {unidad} (MAE) y la desviación "
        f"típica del error es de {rmse:.4f} {unidad} (RMSE)."
    )


def interpretar_coeficiente(
    etiqueta_variable: str,
    estimacion: float,
    unidad_variable: str,
    etiqueta_objetivo: str,
    unidad_objetivo: str,
) -> str:
    """Redacta la interpretación de un coeficiente en unidades del problema."""
    sentido = "aumenta" if estimacion >= 0 else "disminuye"
    return (
        f"Por cada unidad adicional de {etiqueta_variable} ({unidad_variable}), "
        f"{etiqueta_objetivo} {sentido} en promedio {abs(estimacion):.4f} "
        f"{unidad_objetivo}, manteniendo constantes las demás variables."
    )


__all__ = [
    "calcular_metricas",
    "r2_ajustado",
    "metricas_linea_base",
    "evaluar_modelo",
    "tabla_predicciones",
    "resumen_residuos",
    "interpretar_desempeno",
    "interpretar_coeficiente",
]
