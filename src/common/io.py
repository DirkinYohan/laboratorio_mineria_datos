"""Utilidades de entrada y salida.

Centraliza la lectura de datasets, la escritura de resultados, la exportación de
gráficas y el guardado/carga de modelos, para que los tres modelos no repliquen
la misma lógica de ``pandas``, ``joblib`` y ``matplotlib``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .paths import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    VISUALIZATIONS_DIR,
)


# --- Datos --------------------------------------------------------------
def cargar_dataset(ruta: Path | str) -> pd.DataFrame:
    """Lee un dataset CSV en un ``DataFrame``.

    Raises:
        FileNotFoundError: Si el archivo indicado no existe.
    """
    ruta_csv = Path(ruta)
    if not ruta_csv.is_file():
        raise FileNotFoundError(
            f"No se encontró el dataset: {ruta_csv}. "
            "Verifica que el CSV esté en 'data/raw/'."
        )
    return pd.read_csv(ruta_csv)


def cargar_procesado(nombre: str) -> pd.DataFrame:
    """Lee un dataset limpio desde ``data/processed``.

    Raises:
        FileNotFoundError: Si la etapa de limpieza aún no se ha ejecutado.
    """
    ruta = DATA_PROCESSED_DIR / nombre
    if not ruta.is_file():
        raise FileNotFoundError(
            f"Aún no existe '{ruta}'. Ejecuta primero la limpieza del modelo."
        )
    return pd.read_csv(ruta)


def directorio_resultados(clave_modelo: str) -> Path:
    """Devuelve (y crea si falta) la carpeta de resultados de un modelo."""
    destino = RESULTS_DIR / clave_modelo
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def directorio_visualizaciones(clave_modelo: str) -> Path:
    """Devuelve (y crea si falta) la carpeta de gráficas de un modelo."""
    destino = VISUALIZATIONS_DIR / clave_modelo
    destino.mkdir(parents=True, exist_ok=True)
    return destino


# --- Resultados ---------------------------------------------------------
def guardar_json(payload: Mapping[str, Any], clave_modelo: str, nombre: str) -> Path:
    """Serializa un diccionario (métricas, coeficientes...) a JSON."""
    destino = directorio_resultados(clave_modelo) / nombre
    with destino.open("w", encoding="utf-8") as manejador:
        json.dump(payload, manejador, indent=2, ensure_ascii=False, default=str)
    return destino


def guardar_tabla(df: pd.DataFrame, clave_modelo: str, nombre: str) -> Path:
    """Guarda una tabla de resultados (``.csv``) para el informe."""
    destino = directorio_resultados(clave_modelo) / nombre
    df.to_csv(destino, index=False)
    return destino


# --- Visualizaciones ----------------------------------------------------
def ruta_visualizacion(clave_modelo: str, nombre: str) -> Path:
    """Devuelve la ruta destino de una gráfica, creando la carpeta si falta."""
    return directorio_visualizaciones(clave_modelo) / nombre


def guardar_figura(figura: Any, clave_modelo: str, nombre: str, dpi: int = 150) -> Path:
    """Exporta una figura de Matplotlib a ``visualizations/<modelo>/``."""
    destino = ruta_visualizacion(clave_modelo, nombre)
    figura.savefig(destino, dpi=dpi, bbox_inches="tight")
    return destino


# --- Modelos ------------------------------------------------------------
def guardar_modelo(modelo: Any, especificacion: Any) -> Path:
    """Exporta un modelo entrenado con ``joblib`` a ``models/<modelo>/``.

    Args:
        modelo: Estimador ya entrenado.
        especificacion: Especificación del modelo (define la ruta destino).

    Returns:
        La ruta del artefacto escrito.
    """
    import joblib  # Import perezoso: solo se necesita al exportar.

    destino = especificacion.ruta_modelo
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, destino)
    return destino


def cargar_modelo(especificacion: Any) -> Any:
    """Carga el modelo exportado de una especificación.

    Raises:
        FileNotFoundError: Si el artefacto aún no ha sido entrenado.
    """
    import joblib

    origen = especificacion.ruta_modelo
    if not origen.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo exportado: {origen}. "
            f"Ejecuta primero el entrenamiento: "
            f"python -m src.{especificacion.clave}.modelo"
        )
    return joblib.load(origen)


# --- Calidad ------------------------------------------------------------
def contar_nulos(df: pd.DataFrame, columnas: Sequence[str]) -> dict[str, int]:
    """Cuenta valores nulos por columna de interés."""
    return {columna: int(df[columna].isna().sum()) for columna in columnas if columna in df}


__all__ = [
    "cargar_dataset",
    "cargar_procesado",
    "directorio_resultados",
    "directorio_visualizaciones",
    "guardar_json",
    "guardar_tabla",
    "ruta_visualizacion",
    "guardar_figura",
    "guardar_modelo",
    "cargar_modelo",
    "contar_nulos",
]
